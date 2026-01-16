import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from accelerate import Accelerator
from transformers import BertTokenizer
import nltk
import re
from nltk.tokenize import sent_tokenize
from transformer import DataCollatorForLanguageModeling
import random
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import BertForPreTraining, BertConfig

accelerator = Accelerator()

def create_sentence_dataframe(df):
    sentences = []

    special_chars_pattern = re.compile(r'[^a-zA-Z0-9\s.,?!]+|\n')

    for text in df['TEXT']:
        clean_text = special_chars_pattern.sub('', text)

        tokenized_sentences = sent_tokenize(clean_text)

        sentences.extend(tokenized_sentences)

    sentence_df = pd.DataFrame(sentences, columns = ['text'])

    return sentence_df

data_txt = pd.read_csv("archive/medical_data.csv")
data = create_sentence_dataframe(data_txt)

class ClinizalDataset(Dataset):
    def __init__(self, data, tokenizer, max_length = 512):
        self.data = dataself.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        news = self.data.loc[idx, 'text']
        if idx + 1 < len(self.data):
            next_news = self.data.loc[idx + 1, 'text']
        else:
            next_news =  self.data.loc[0, 'text']

        combined_news = news + " [SEP] " + next_news
        tokenized = self.tokenizer(combined_news, truncation = True, padding = 'max_length',
                                   max_length = self.max_length, return_tensor = 'pt')
        return {'input_ids' : tokenized['input_ids'].squeeze(0),
                'attention_mask' : tokenized['attention_mask'].squeeze(0),
                'text' : combined_news}

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dataset = ClinizalDataset(data, tokenizer)

class DataCollatorForPreTraining(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm = True, mlm_probability = 0.15, nsp_probability = 0.5):
        super().__init__(tokenizer = tokenizer, mlm = mlm, mlm_probability = mlm_probability)
        self.nsp_probability = nsp_probability

    def __call__(self, examples):
        nsp_labels = []
        input_ids_list = []
        attention_masks_list = []
        labels_list = []

        for example in examples:
            input_ids = example['input_ids']
            attention_mask = example['attention_mask']

            if random.random() > self.nsp_probability:
                nsp_labels.append(1)
            else:
                nsp_labels.append(0)

                sep_idx = (input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple = True)[0][0].item()
                second_sentence = input_ids[sep_idx + 1:]
                second_sentence = second_sentence[torch.randperm(second_sentence.size()[0])]

                input_ids = torch.cat((input_ids[:sep_idx + 1], second_sentence), dim = 0)

            input_ids_list.append(input_ids)
            attention_masks_list.append(attention_mask)

            sep_idx = (input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple = True)[0][0].item()
            labels = input_ids.clone()
            labels[sep_idx:] = -100
            labels_list.append(labels)

        example_dicts = [{'input_ids' : ids, 'attention_mask' : mask, 'labels' : lbl} for ids, mask, lbl in zip(input_ids_list, attention_masks_list, labels_list)]

        batch = super().__call__(example_dicts)

        batch['next_sentence_label'] = torch.tensor(nsp_labels, dtype = torch.long)
        return batch

data_collator = DataCollatorForPreTraining(tokenizer)
train_dataloader = DataLoader(    dataset, shuffle=True, collate_fn=data_collator, batch_size=16)

config = BertConfig.from_pretrained('bert-base-uncased')
model = BertForPreTraining(config)

optimizer = AdamW(model.parameters(), lr = 5e-5)

model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

num_epochs = 1
print_every = 10

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    model.train()
    running_loss = 0.0

    for step, batch in enumerate(train_dataloader):
        input_ids = batch['input_ids'].to(accelerator.device)
        attention_mask = batch['attention_mask'].to(accelerator.device)
        labels = batch['labels'].to(accelerator.device)
        next_sentence_label = batch['next_sentence_label'].to(accelerator.device)

        outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels, next_sentence_label = next_sentence_label)
        loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

        if (step + 1) % print_every == 0:
            print(f'Step {step + 1}: Loss = {running_loss / print_every:.4f}')
            running_loss = 0.0

print('Finished Training')