from PIL import Image
import os
jpeg_directory = './data/images/train'

for filename in os.listdir(jpeg_directory):
    if filename.lower().endswith('.jpg.') or filename.lower().endswith('.jpeg'):
        jpeg_file_path = os.path.join(jpeg_directory, filename)

        with Image.open(jpeg_file_path) as image:
            png_file_path = os.path.join(jpeg_file_path, filename.rsplit('.',1)[0] + '.png')
            image.save(png_file_path, 'PNG')

        os.remove(jpeg_file_path)

from dataclasses import dataclass

@dataclass
class TrainingConfiguration:
    image_size = 128
    train_batch_size = 4
    eval_batch_size = 32
    num_epochs = 100
    gradient_accmulation_steps = 2
    learning_rate = 2e-4
    lr_warmup_steps = 100
    save_image_epochs = 4
    save_model_epochs = 4
    mixed_precision = 'fp16'
    output_dir = './image_gen1'

    overwrite_output_dir = True
    seed = 100
config = TrainingConfiguration()

import torch
if not torch.cuda.is_available():
    config.mixed_precision = False

from datasets import load_dataset

dataset = load_dataset("BirdL/DALL-E-Dogs", split="train")

import matplotlib.pyplot as plt

def plot_images(images, titles = None, cols = 2, figsize = (10, 5)):
    rows = len(images) // cols + int(len(images) % cols > 0)
    fig, axs = plt.subplots(rows, cols, figsize = figsize)
    axs = axs.flatten()

    for idx, img in enumerate(images):
        axs[idx].imshow(img)
        axs[idx].axis('off')
        if titles:
            axs[idx].set_title(titles[idx])

    plt.tight_layout()
    plt.show()

four_images = [dataset[i]['Images'] for i in range(4)]

plot_images(four_images)

from torchvision import transforms

preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees = 15),
        transforms.ColorJitter(brightness = 0.1, contrast = 0.1, saturation = 0.1, hue = 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]
)

def transform(examples):
    images = [preprocess(image.convert('RGB')) for image in examples['Images']]
    return {'image' : images}

dataset.set_transform(transform)

fig, axes = plt.subplots(2, 2, figsize = (8, 8))
for i in range(2):
    for j in range(2):
        index = i * 2 + j
        image = dataset[index]['image'].permute(1, 2, 0).numpy() * 0.5 + 0.5
        axes[i, j].imshow(image)
        axes[i, j].axis('off')

plt.show()

import torch

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size = config.train_batch_size, shuffle = True)

from diffusers import UNet2DModel

model = UNet2DModel(
    sample_size = config.image_size,
    in_channels = 3,
    out_channels = 3,
    layers_per_block = 3,
    block_out_channels = (64, 128, 256, 512, 1024),
    down_block_types = (
        'DownBlock2D',
        'AttnDownBlock2D',
        'DownBlock2D',
        'DownBlock2D',
        'DownBlock2D'
    ),
    up_block_types = (
        'UpBlock2D',
        'UpBlock2D',
        'AttnUpBlock2D',
        'UpBlock2D',
        'UpBlock2D'
    )
)

from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(num_train_timesteps = 1000)

sample_image = dataset[0]['image'].unsqueeze(0)

noise_tensor = torch.randn(sample_image.shape)
num_timesteps = torch.LongTensor([50])
perturbed_image = noise_scheduler.add_noise(sample_image, noise_tensor, num_timesteps)

restored_image = ((perturbed_image.permute(0, 2, 3, 1) + 1.0) * 140).type(torch.uint8).numpy()[0]
output_image = Image.fromarray(restored_image)

import torch.nn.functional as F
optimizer = torch.optim.AdamW(model.parameters(), lr = config.learning_rate)

from diffusers.optimization import get_cosine_schedule_with_warmup

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer = optimizer,
    num_warmup_steps = config.lr_warmup_steps,
    num_training_steps = (len(train_dataloader) * config.num_epochs)
)

import math

def create_image_grid(images, num_rows, num_cols):
    width, height = image[0].size
    grid = Image.new('RGB', size = (num_cols * width, num_rows * height))
    for idx, img in enumerate(images):
        grid.paste(img, box = (idx % num_cols, width, idx // num_cols * height))
    return grid

def perform_evaluation(updated_config, current_epoch, diffusion_pipeline):
    generated_images = diffusion_pipeline(
        batch_size = updated_config.eval_batch_size,
        generator = torch.manual_seed(updated_config.seed)
    ).images

    img_grid = create_image_grid(generated_images, num_rows = 4, num_cols = 4)

    test_output_dir = os.path.join(updated_config.output_dir, 'samples')
    os.makedir(test_output_dir, exist_ok = True)
    img_grid.save(f'{test_output_dir}/{current_epoch:04d}.png')

from accelerate import Accelerator
from tqdm.auto import tqdm

def train(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision = 'no',
        gradient_accmulation_steps = config.gradient_accmulation_steps,
        log_with = 'tensorboard',
        project_dir = os.path.join(config.output_dir, 'logs')
    )

    if config.output_dir is not None:
        os.makedir(config.output_dir, exist_ok = True)
        accelerator.init_trackers('train_example')

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    overall_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total = len(train_dataloader), disable = not accelerator.is_local_main_process)
        progress_bar.set_description(f'Epoch {epoch}')

        for step, batch in enumerate(train_dataloader):
            clean_images = batch['image']
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            timesteps = torch.randint(0, noisr_scheduler.num_train_timesteps, (bs,), device = clean_images.device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accmulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict = False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {'loss' : loss.detach().item(), 'lr' : lr_scheduler.get_last_lr()[0], 'step' : overall_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step = overall_step)
            overall_step += 1

        if accelerator.is_main_process():
            pipeline = DDPMPipeline(unet = accelerator.unwrap_model(model), scheduler = noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                perform_evaluation(config, epoch, pipeline)
                pipeline.save_pretrained(config.output_dir)


train(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

from diffusers import DDIMPipeline

model_id = './image_gen1/'
ddim = DDIMPipeline.from_pretrained(model_id).to('cuda')

image = ddim().images[0]
image.show()

generated_images = ddim(batch_size = 6).images

num_rows = 2
num_cols = 3

width, height = generated_images[0].size
grid = Image.new('RGB', size = (num_cols * width, num_rows * height))

grid.show()
