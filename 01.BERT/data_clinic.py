import pandas as pd
from sklearn.model_selection import train_test_split
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler

#데이터 전처리
real = pd.read_csv('/archive/True.csv')
fake = pd.read_csv('/archive/False.csv')

real = real.drop(['title', 'subject', 'date'], axis = 1)
real['label'] = 1.0
fake = fake.drop(['title', 'subject', 'date'], axis = 1)
fake['label'] = 0.0

dataframe = pd.concat([real, fake], axis = 0, ignore_index = True)

#토크나이저
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

#데이터셋 준비하기
data = list(zip(df['text'].tolist(), df['label'].tolist()))

def tokenize_and_encode(texts, labels):
    input_ids, attention_masks, labels_out = [], [], []
    for text, label in zip(texts, labels):
        encoded = tokenizer.encode_plus(
            text, max_length = 512, padding = 'max_length', truncation = True)
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        labels_out.append(label)
    return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(labels_out)

texts, labels = zip(*data)

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size = 0.2)

train_input_ids, train_attention_masks, train_labels = tokenize_and_encode(train_texts, train_labels)

val_input_ids, val_attention_masks, val_labels = tokenize_and_encode(val_texts, val_labels)

#커스텀 데이터셋 클래스
class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks, labels, num_classes = 2):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.num_classes = num_classes
        self.one_hot_labels = self.one_hot_encode(labels, num_classes)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids' : self.input_ids[idx],
            'attention_masks' : self.attention_masks[idx],
            'labels' : self.one_hot_labels[idx]
        }

    @staticmethod
    def one_hot_encode(targets, num_classes):
        targets = targets.long()
        one_hot_targets = torch.zeros(targets.size(0), num_classes)
        one_hot_targets.scatter_(1, targets.unsqueeze(1), 1.0)
        return one_hot_targets

train_dataset = TextClassificationDataset(
    train_input_ids, train_attention_masks, train_labels)

val_dataset = TextClassificationDataset(
    val_input_ids, val_attention_masks, val_labels)

#DataLoader 생성
train_dataloader = DataLoader(train_dataset, batch_size = 8, shuffle = True)
eval_dataloader = DataLoader(val_dataset, batch_size = 8)

#점검
item = next(iter(train_dataloader))
item_ids, item_mask, item_labels = item['input_ids'], item['attention_masks'], item['labels']
print('item_ids', item_ids.shape)
print('item_mask', item_mask.shape)
print('item_labels', item_labels.shape)

#BERT 불러오기
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 2)
optimizer = AdamW(model.parameters(), lr = 5e-5)

#Accelerator 설정
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader)

#파인튜닝
num_epochs = 1
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer = optimizer,
    num_warmup_steps = 0,
    num_training_steps = num_training_steps)
progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    model.eval()
    device = accelerator.device
    preds = []
    out_label_ids = []
    epochs = 1
    epoch = 1

    for batch in eval_dataloader:
        with torch.no_grad():
            inputs = {k : v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            logits = outputs.logits

        preds.extend(torch.argmax(logits.detach().cpu(), dim = 1).numpy())
        out_label_ids.extend(torch.argmax(inputs["labels"].detach().cpu(), dim = 1).numpy())
    accuracy = accuracy_score(out_label_ids, preds)
    f1 = f1_score(out_label_ids, preds, average = 'weighted')
    recall = recall_score(out_label_ids, preds, average = 'weighted')
    precision = precision_score(out_label_ids, preds, average = 'weighted')

#추론
def inference(text, model, label, device = device):
    inputs = tokenizer(text, return_tensors = 'pt', padding = True, truncation = True)
    inputs = {k : v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    pred_label_idx = torch.argmax(logits.detach().cpu(), dim = 1).item()
    return pred_label_idx