import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
import nltk
from transformers import BertTokenizer, BertModel
from transformers import AlbertModel
from accelerate import Accelerator
from tqdm import tqdm

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

train_dataset = IMDBDataset(imdb_data['train'], tokenizer)
test_dataset = IMDBDataset(imdb_data['test'], tokenizer)

train_5_percent = int(len(train_dataset) * 0.05)
test_5_percent = int(len(test_dataset) * 0.05)

train_95_percent = len(train_dataset) - train_5_percent
test_95_percent = len(test_dataset) - test_5_percent

def pad_collate(batch):
    max_num_sentences = max([item['input_ids'].shape[0] for item in batch])

    input_ids_batch = []
    attention_masks_batch = []

    for item in batch:
        num_sentences = item['input_ids'].shape[0]
        pad_length = max_num_sentences - num_sentences
        input_ids = torch.cat([item['input_ids'], torch.zeros(pad_length, item['input_ids'].shape[1], dtype = torch.long)], dim = 0)
        attention_mask = torch.cat([item['attention_mask'], torch.zeros(pad_length, item['attention_mask'].shape[1], dtype = torch.long)], dim = 0)

        input_ids_batch.append(input_ids)
        attention_masks_batch.append(attention_mask)

    input_ids_tensor = torch.stack(input_ids_batch, dim = 0)
    attetion_masks_tensor = torch.stack(attention_masks_batch, dim = 0)
    labels_tensor = torch.tensor([item['label'] for item in batch], dtype = torch.long)

    return {'input_ids' : input_ids_tensor, 'attention_mask' : attetion_masks_tensor, 'labels' : labels_tensor}

train_loader = DataLoader(train_dataset, batch_size = 2, shuffle = True, collate_fn = pad_collate)
test_loader = DataLoader(test_dataset, batch_size = 2, shuffle = True, collate_fn = pad_collate)

class AlbertTextClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AlbertTextClassifier, self).__init__()
        self.albert = AlbertModel.from_pretrained('albert-base-v2')
        self.attention = nn.Linear(self.albert.config.hidden_size, 1)
        self.classifier = nn.Linear(self.albert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        batch_size, num_sentences, seq_len = input_ids.shape
        input_ids = input_ids.view(batch_size * num_sentences, seq_len)
        attention_mask = attention_mask.view(batch_size * num_sentences, seq_len)

        outputs = self.albert(input_ids, attention_mask = attention_mask)
        hidden_states = outputs.last_hidden_state
        attention_weights = torch.softmax(
            self.attention(hidden_states), dim = 1)
        sentence_representation = torch.sum(attention_weights * hidden_states, dim = 1)
        sentence_representation = sentence_representation.view(batch_size, num_sentences, -1)

        doc_attention_weights = torch.softmax(self.attention(sentence_representation), dim = 1)
        doc_representation = torch.sum(doc_attention_weights * sentence_representation, dim = 1)
        doc_representation = self.dropout(doc_representation)

        logits = self.classifier(doc_representation)
        return logits

acclelerator = Accelerator()
device = acclelerator.device

model = AlbertTextClassifier(num_classes = 2)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

model, optimizer = accelerator.prepare(model, optimizer)

criterion = nn.CrossEntropyLoss()

train_loader, test_loader = accelerator.prepare(train_loader, test_loader)
save_interval = 100

num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    train_loader_progress = tdqm(enumerate(train_loader), desc = f"Epoch {epoch + 1}/{num_epochs}, Training", total=len(train_loader))
    for batch_idx, batch in train_loader_progress:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        if batch_idx % 10 == 0:
            train_loader_progress.set_postfix(loss = loss.item())
        if (batch_idx + 1) % save_interval == 0:
            model_save_path = f'AlbertTextClassifier_epoch{epoch + 1}_batch{batch_idx + 1}.pt'
            torch.save(model.state_dict(), model_save_path)
    model.eval()
    total_correct = 0
    total_samples = 0
    test_loader_progress = tqdm(test_loader, desc = f"Epoch {epoch + 1}/{num_epochs}, Testing")

    with torch.no_grad():
        for batch in test_loader_progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            _, preds = torch.max(logits, dim = 1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
