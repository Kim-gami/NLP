import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from timm import create_model
from transformers import default_data_collator
from datasets import load_dataset, DatasetDict, load_from_disk, Dataset
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def process_image(image_file, image_directory, label_directory):
    file_name, _ = os.path.splitext(image_file)
    label_file = f'{file_name}.png'

    with Image.open(os.path.join(image_directory, image_file)) as im:
        img = im.copy()

    with Image.open(os.path.join(label_directory, label_file)) as im:
        lbl = im.convert('L').copy()

    return {'pixel_values' : img, 'label' : lbl}

def load_images_and_labels(image_directory, label_directory):
    image_files = sorted([f for f in os.listdir(image_directory) if f.endswith('.jpg')])
    label_files = sorted([f for f in os.listdir(label_directory) if f.endswith('.png')])

    data = []

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_image, image_files, [image_directory] * len(image_files), [label_directory] * len(image_files))
        for result in results:
            data.append(result)

    return data

def create_image_segmentation_dataset(image_directory, label_directory):
    data = load_images_and_labels(image_directory, label_directory)
    dataset = Dataset.from_dict({"pixel_values": [item["pixel_values"] for item in data], "label": [item["label"] for item in data]})
    return dataset

image_directory = './FoodSeg103/Images/img_dir/train'
label_directory = './FoodSeg103/Images/ann_dir/train'
output_path = './train_dataset.hf'

train_dataset = create_image_segmentation_dataset(image_directory, label_directory)
train_dataset.save_to_disk(output_path)
print(f'Dataset saved to {output_path}')

train_dataset = load_from_disk(output_path)

train_test_split_ratio = 0.7
train_size = int(train_test_split_ratio * len(train_dataset))
test_size = len(train_dataset) - train_size
split_ds = train_dataset.train_test_split(train_size=train_size, test_size=test_size, seed = 42)
final_ds = DatasetDict({'train' : split_ds['train'], 'test' : split_ds['test']})
train_ds = split_ds['train']
test_ds = split_ds['test']

base_dir = './FoodSeg103/'

id2label = {}

with open(base_dir + 'category_id.txt', 'r') as file:
    for line in file:
        id_, label = line.strip().split('\t')
        id2label[int(id_)] = label
label2id = {v : k  for k, v in id2label.items()}
num_labels = len(id2label)

import matplotlib.pyplot as plt

data = train_ds[0]
fir, (ax1, ax2) = plt.subplot(1, 2, figsize = (12, 6))
ax1.imshow(data['pixel_values'])
ax1.set_title('Image')
ax1.axis('off')

ax2.imshow(data['label'], cmap = 'gray')
ax2.set_title('Segmentation Label')
ax2.axis('off')

plt.show()

label = train_ds[0]['label']
import numpy as np
r_channel_array = np.array(label)

unique_categories = np.unique(r_channel_array)

print('Unique categories in the image : ', unique_categories)

from torchvision.transforms import RandomHorizontalFlip, RandomRotation, Compose
from transformers import SegformerFeatureExtractor
from torchvision.transforms import ColorJitter

feature_extractor = SegformerFeatureExtractor()
jitter = ColorJitter(brightness = 0.25, contrast = 0.25, saturation = 0.25, hue = 0.1)

def train_transforms(example_batch):
    images = [jitter(x) for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = feature_extractor(images, labels)
    return inputs

def val_transforms(example_batch):
    images = [x for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = feature_extractor(images, labels)
    return inputs

train_ds.set_transforms(train_transforms)
test_ds.set_transforms(val_transforms)

from transformers import SegformerForSemanticSegmentation

model_name = 'nvidia/mit-b0'
model = SegformerForSemanticSegmentation.from_pretrained(model_name,
                                                         id2label=id2label,
                                                         label2id=label2id)

from transformers import TrainArguments

epochs = 10
lr = 3e-5
batch_size = 4

training_args = TrainArguments(
    '/run',
    learning_rate = lr,
    num_train_epochs = epochs,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    save_total_limit = 5,
    evaluation_strategy = 'steps',
    save_strategy = 'steps',
    save_steps = 10000,
    eval_steps = 1000,
    logging_steps = 500,
    eval_accumulation_steps = 10,
    load_best_model_at_end = True)

import evaluate

iou_metric = evaluate.load('mean_iou')

def caculate_segmentation_metrics(prediction_ground_truth):
    with torch.no_grad():
        logits, ground_truth = prediction_ground_truth
        logits_as_tensor = torch.from_numpy(logits)

        resized_logits = nn.functional.interpolate(
            logits_as_tensor,
            size = ground_truth.shape[-2:],
            mode = 'bilinear',
            align_corners = False
        ).argmax(dim = 1)

        predicted_labels = resized_logits.detach().cpu().numpy()

        segmentation_metrics = iou_metric.compute(
            predictions = predicted_labels,
            references = ground_truth,
            num_labels = len(id2label),
            ignore_index = 0,
            reduce_labels = feature_extractor.do_reduce_labels
        )

        category_accuracy = segmentation_metrics.pop('per_category_accuracy').tolist()
        category_iou = segmentation_metrics.pop('per_category_iou').tolist()

        segmentation_metrics.updata({f'accuracy_{id2label[i]}' : v for i, v in enumerate(category_accuracy)})
        segmentation_metrics.update({f'iou_{id2label[i]}' : v for i, v in enumerate(category_iou)})

    return segmentation_metrics

from transformers import Trainer

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_ds,
    eval_dataset = test_ds,
    compute_metrics = caculate_segmentation_metrics
)

import wandb
wandb.init(project = 'segmentation_project', name = 'run)name')

trainer.train()

save_directory = './model/'
model.save_pretrained(save_directory)

load_directory = save_directory
loaded_model = SegformerForSemanticSegmentation.from_pretrained(
    load_directory,
    id2label = id2label,
    label2id = label2id)

import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from transformers import SegformerFeatureExtractor
from matplotlib import pyplot as plt

feature_extractor = SegformerFeatureExtractor()

image = Image.open('./FoodSeg103/Images/img_dir/train/00000000.jpg')

inputs = feature_extractor(imgae = [image], return_tensors = 'pt')

with torch.no_grad():
    outputs = loaded_model(**inputs)
    predictions = outputs.logits.argmax(dim = 1).squeeze().cpu().numpy()

grayscale_map = np.zeros((predictions.shape[0], predictions.shape[1]), dtype = np.uint8)
for label_id in id2label.keys():
    grayscale_map[predictions == label_id] = label_id

    segmentation_image = Image.fromarray(grayscale_map, mode = 'L')

    true_label_path = './FoodSeg103/Images/ann_dir/train/00000000.png'
    true_label = Image.open(true_label_path).convert('L')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 5))
    ax1.inshow(image)
    ax1.set_title('Original Image')
    ax2.imshow(segmentation_image, cmap = 'gray')
    ax2.set_title('Predicted Segmentation Map')
    ax3.imshow(true_label, cmap = 'gray')
    ax3.set_title('True Label')
    plt.show()