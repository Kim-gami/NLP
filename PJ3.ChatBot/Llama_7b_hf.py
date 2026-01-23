import xmltodict
import json
import glob
import os
from huggingface_hub import login
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from peft import prepare_model_for_kbit_training
import transformers
from peft import PeftModel, PeftConfig
import textwrap
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from transformers.generation.utils import GreedySearchDecoderOnlyOutput

base_directory = './MedQuAD-master'

def convert_xml_to_json(xml_file):
    with open(xml_file, 'r', encoding = 'utf-8') as f:
        xml_data = f.read()

    xml_dict = xmltodict.parse(xml_data)

    if 'Document' not in xml_dict or xml_dict['Document'] in None or 'QAPairs' not in xml_dict['Document'] or xml_dict['Document']['QAPairs'] is None:
        return []

    questions = xml_dict['Document']['QAPairs']['QAPair']

    if not isinstance(questions, list):
        questions = [questions]

    json_data = []

    for question in questions:
        if question['Answer'] and question['Answer'].strip():
            json_entry = {
                'instruction' : question['Question']['#text'],
                'input' : '',
                'output' : question['Answer']
            }
            json_data.append(json_entry)
    return json_data

file_path = base_directory

combined_json_data = []

for root, dirs, files in os.walk(file_path):
    for file in files:
        if file.endswith('.xml'):
            xml_file_path = os.path.join(root, file)
            combined_json_data.extend(convert_xml_to_json(xml_file_path))

output_file = os.path.join(base_directory, 'alpaca_data.json')
with open(output_file, 'w', encoding = 'utf-8') as f:
    json.dump(combined_json_data, f, indent = 4, ensure_ascii = False)

login()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_MODEL = "baffo32/decapoda-research-llama-7B-hf"

model = LlamaForCausalLM.from_pretrained(BASE_MODEL, load_in_8bit = True, device_map = 'auto')

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

tokenizer.pad_token_id = 0
tokenizer.padding_side = 'left'

train_data = load_dataset("json", data_files=base_directory+"alpaca_data.json", split="train[:90%]")
valid_data = load_dataset("json", data_files=base_directory+"alpaca_data.json", split="train[90%:]")

def generate_prompt(data_point):
    if data_point['input']:
        return f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
{data_point['instruction']}

{data_point['input']}

{data_point['output']}'''
    else:
        return f'''Below is an instruction that describes a task. Write a response that appropriately completes the request.

{data_point['instruction']}

{data_point['input']}

{data_point['output']}'''

data_train = train_data.shuffle().map(
    lambda data_point : tokenizer(
        generate_prompt(data_point),
        truncation = True,
        max_length = 1000,
        padding = 'max_length'
    )
)
data_valid = valid_data.shuffle().map(
    lambda data_point : tokenizer(
        generate_prompt(data_point),
        truncation = True,
        max_length = 1000,
        padding = 'max_length'
    )
)

data_train_save_path = os.path.join(base_directory, 'data_train_saved.json')
data_valid_save_path = os.path.join(base_directory, 'data_valid_saved.json')

data_train.save_to_disk(data_train_save_path)
data_valid.save_to_disk(data_valid_save_path)

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

LORA_TARGET_MODULES = [
    'q_proj',
    'v_proj'
]

BATCH_SIZE = 128
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 4e-4
TRAIN_STEPS = 50
OUTPUT_DIR = base_directory

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r = LORA_R,
    lora_alpha = LORA_ALPHA,
    target_module = LORA_TARGET_MODULES,
    lora_dropout = LORA_DROPOUT,
    bias = 'none',
    task_type = 'CAUSAL_LM'
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size = MICRO_BATCH_SIZE,
    gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
    warmup_steps = 10,
    max_steps = TRAIN_STEPS,
    learning_rate = LEARNING_RATE,
    fp16 = True,
    logging_steps = 10,
    optim = 'adamw_torch',
    evaluation_strategy = 'steps',
    save_strategy = 'steps',
    eval_steps = 50,
    output_dir = OUTPUT_DIR,
    save_total_limit = 3,
    load_best_model_at_end = True,
    report_to = 'tensorboard'
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data_train,
    eval_dataset=data_valid,
    args=training_arguments,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train(resume_from_checkpoint=False)

model.save_pretrained(base_directory+"chatbot")

peft_model_id = base_directory + 'chatbot'

config = PeftConfig.from_pretrained(peft_model_id)

model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype = torch.float16,
    device_map = 'auto'
)

model = PeftModel.from_pretrained(
    model,
    peft_model_id,
    torch_dtype = torch.float16,
    offload_buffers = True
)

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

if torch.cuda.is_available():
    model = model.to('cuda')
else:
    model = model.to('cpu')

model.eval()

input_text = 'Hello, how can I assist you today?'
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)

outputs = model.generate(input_ids, max_length = 50)
print(tokenizer.decode(outputs[0], skip_special_tokens = True))

def ask_ai_doctor(instruction : str, model : PeftModel) -> str:
    PROMPT_TEMPLATE = f'''
    Below is an instruction that describes a task. Write a response that appropriately completes the request

    ### Instruction:
    [INSTRUCTION]

    ### Response:
    '''

    prompt = PROMPT_TEMPLATE.replace('[INSTRUCTION]', instruction)

    encoding = tokenizer(prompt, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)

    '''
    controls various aspects of the text generation process.
    temperature: This parameter (set to 0.1) controls the randomness of the generated text. lower value more determenistic; higher value more random
    top_p: This parameter (set to 0.75) is also called nucleus sampling. In our case, the model will only consider tokens that make up the top 75% of probabilities for the next word
    repetition_penalty: This parameter (set to 1.1) is used to penalize repetitions in the generated text. A value greater than 1 helps to reduce the frequency of repeated phrases
    '''
    generation_config = GenerationConfig(
        temperature = 0.1,
        top_p = 0.75,
        repetition_penalty = 1.1
    )

    with torch.inference_mode():
        response = model.generate(
            input_ids = input_ids,
            generation_config = generation_config,
            return_dict_in_generate = True,
            output_scores = True,
            max_new_tokens = 250
        )

    decoded_output = tokenizer.decode(response.sequences[0])
    formatted_response = decoded_output.split('### Response:')[1].strip()

    return '\n'.join(textwrap.wrap(formatted_response))

print(ask_ai_doctor('What are symptoms of Cirrhosis?', model))