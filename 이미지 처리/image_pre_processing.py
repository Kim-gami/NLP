import torch
import torchvision.transforms as T
from PIL import Image
import requests
from io import BytesIO

img_path = './tulip_field.png'

img = Image.open(img_path)

img = img.convert('RGB')

# img.show()

transforms = T.Compose([
    T.RandomRotation(degrees = (-15, 15), fill = 0),
    T.RandomResizedCrop(size = (224, 224), scale = (0.8, 1.0)),
    T.RandomHorizontalFlip(p = 0.5),
    T.RandomVerticalFlip(p = 0.5),
    T.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.1),
    T.ToTensor(),
    T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

augmented_img = transforms(img)

unnormalized_img = T.Compose([
    T.Normalize(mean = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
                std = [1 / 0.229, 1 / 0.224, 1 / 0.255])(augmented_img)
    