import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
import nltk
from transformers import BertTokenizer, BertModel

imdb_data = load_dataset('imdb')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class IMDBDataset(Dataset):
    def __init__(self, data, tokenizer, max_sentence_length = 48):
        self.data = data
        self.tokenizer = tokenizer
        self.max_sentence_length = max_sentence_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = self.data[idx]['label']
        sentences = nltk.sent_tokenize(text)
        input_ids = [tokenizer.encode(sentence, max_length = self.max_sentence_length, truncation = True, padding = 'max_length') for sentence in sentences]
        attention_masks = [[1 if token_id != tokenizer.pad_token_id else 0 for token_id in sentence] for sentence in input_ids]
        return {'input_ids' : torch.tensor(input_ids, dtype = torch.long),
                'attention_mask' : torch.tensor(attention_masks, dtype = torch.long),
                'label' : torch.tensor(label, dtype = torch.long)}