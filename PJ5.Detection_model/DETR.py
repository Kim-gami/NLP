import torch
import torchvision.transforms as T
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from transformers import DetrForObjectDetection, DetrImageProcessor, DetrConfig

img_path = './tulip_field.png'
img = Image.open(img_path)
img = img.convert('RGB')

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img_tensor = transform(img).unsqueeze(0)

config = DetrConfig.from_pretrained('facebook/detr-resnet-50')

processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')

model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50', config = config)

model.eval()

with torch.no_grad():
    outputs = model(img_tensor)

target_sizes = torch.tensor([img.size[::-1]])
results = processor.post_process_object_detection(
    outputs, target_sizes = target_sizes, threshold = 0.9)[0]

fig, ax = plt.subplots(1, 1, figsize = (10, 10))
ax.imshow(img)

colors = plt.get_cmap('tab20').colors

for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
    x, y, w, h = box
    w = w - x
    h = h - y
    rect = plt.Rectangle((x, y), w, h, linewidth = 1, edgecolor = colors[label % 20], facecolor = 'none')
    ax.add_patch(rect)
    ax.text(
        x, y,
        f'{model.config.id2label[label.item()]}\n'
        f'{round(score.item(), 3)}',
        fontsize = 15,
        color = colors[label % 20]
    )
plt.show()