import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from transformers import GPT2LMHeadModel
from torch.optim import AdamW
from tqdm import tqdm

model_name = 'gpt2'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

def split_text(text, max_length = 100):
    return [text[i : i + max_length] for i in range(0, len(text), max_length)]

split_texts = split_text(text)

tokenized_texts = tokenizer(split_texts, return_tensors = 'pt', padding = True, truncation = True)

class ShiftDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        input_ids = self.encodings['input_ids'][idx]
        attention_mask = self.encodings['attention_mask'][idx]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {'input_ids' : input_ids,
                'attention_mask' : attention_mask,
                'labels' : labels}

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = ShiftDataset(tokenized_texts)
train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size = 4)

accelerator = Accelerator()

num_epochs = 10
lr = 5e-5

model = GPT2LMHeadModel.from_pretrained('gpt2')
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

model, optimizer = accelerator.prepare(model, optimizer)
train_dataloader = accelerator.prepare(train_dataloader)

for epoch in range(num_epochs):
    epoch_iterator = tqdm(train_dataloader, desc = f'Epoch {epoch + 1}')
    for step, batch in enumerate(epoch_iterator):
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
        loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()

        if step % 500 == 0:
            epoch_iterator.set_postfix({'Loss' : loss.item()}, refresh = True)

    if (epoch + 1) % 5 == 0:
        model_save_path = f'./{epoch + 1}'
        model.save_pretrained(model_save_path)

