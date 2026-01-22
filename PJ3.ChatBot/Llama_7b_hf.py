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

base_directory = './MedQuAD-master'

def convert_xml_to_json(xml_file):
    with open(xml_file, 'r', encoding = 'utf-8') as f:
        xml_data = f.read()

    xml_dict = xmltodict.parse(xml_data)

    if 'Document' not in xml_dict or xml_dict['Document'] in None or 'QAPairs' not in xml_dict['Document'] or xml_dict['Document']['QAPairs'] is None:
        return []

    questions = xml_dict['Document']['QAPairs']['QAPair']

    if not isinstance(question, list):
        question = [question]

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

train_data1 = load_dataset('json', data_files = base_directory+"alpaca_data.json", split="train[:90%]")
valid_data2 = load_dataset('json', data_files = base_directory+"alpaca_data.json", split="train[90%:]")
data_train = train_data1.shuffle().map(
    lambda data_point : tokenizer(
        generate_prompt(data_point),
        truncation = True,
        max_length = 1000,
        padding = 'max_length'
    )
)
data_valid = valid_data2.shuffle().map(
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