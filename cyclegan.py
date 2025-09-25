#!/usr/bin/env python3
# cyclegan.py
# Full minimal CycleGAN implementation (ResNet generator + PatchGAN discriminator)
# Tested with PyTorch 1.10+ style API.
# python cyclegan.py --dataset_name anime2pokemon --n_epochs 200

import argparse
import itertools
import os
import random
from tqdm import tqdm

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torchvision.transforms import InterpolationMode


# ---------------------------
# Options / CLI
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="celebs2anime", help="name of the dataset folder")
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: beta1")
parser.add_argument("--b2", type=float, default=0.999, help="adam: beta2")
parser.add_argument("--decay_epoch", type=int, default=10, help="epoch to start lr decay")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
parser.add_argument("--save_interval", type=int, default=1000, help="save image every N batches")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--seed", type=int, default=42)
opt = parser.parse_args()

# reproducibility
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

# ---------------------------
# Dataset
# ---------------------------
import glob

class ImageDataset(Dataset):
    """
    Robust CycleGAN dataset loader.
    Expects:
        root/trainA/, root/trainB/, root/testA/, root/testB/
    Supports multiple image extensions.
    Returns: dict {"A": imgA_tensor, "B": imgB_tensor}
    """
    IMG_EXTS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]

    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms_

        # Collect all images for domain A
        self.files_A = sorted(list(itertools.chain.from_iterable(
            glob.glob(os.path.join(root, f"{mode}A", ext)) for ext in self.IMG_EXTS
        )))
        # Collect all images for domain B
        self.files_B = sorted(list(itertools.chain.from_iterable(
            glob.glob(os.path.join(root, f"{mode}B", ext)) for ext in self.IMG_EXTS
        )))

        # Check for empty folders
        if len(self.files_A) == 0:
            raise RuntimeError(f"No images found in {os.path.join(root, mode+'A')}")
        if len(self.files_B) == 0:
            raise RuntimeError(f"No images found in {os.path.join(root, mode+'B')}")

        print(f"[INFO] {mode}A images: {len(self.files_A)}")
        print(f"[INFO] {mode}B images: {len(self.files_B)}")

    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index % len(self.files_A)]).convert("RGB")
        img_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert("RGB")

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


transform = transforms.Compose([
    transforms.Resize((opt.img_height, opt.img_width), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])



dataset_root = os.path.join("data", opt.dataset_name)
train_dataset = ImageDataset(dataset_root, transforms_=transform, mode="train")
val_dataset = ImageDataset(dataset_root, transforms_=transform, mode="test")


# Debug prints
print(f"trainA images: {len(train_dataset.files_A)}")
print(f"trainB images: {len(train_dataset.files_B)}")
print(f"valA images: {len(val_dataset.files_A)}")
print(f"valB images: {len(val_dataset.files_B)}")

dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
val_dataloader = DataLoader(val_dataset, batch_size=5, shuffle=True, num_workers=1)

# ---------------------------
# Networks
# ---------------------------
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

# Residual block used in generator
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)

# ResNet-based generator
class GeneratorResNet(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super().__init__()

        # Initial conv block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        curr_dim = 64
        for _ in range(2):
            model += [
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(curr_dim * 2),
                nn.ReLU(inplace=True)
            ]
            curr_dim *= 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(curr_dim)]

        # Upsampling
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(curr_dim // 2),
                nn.ReLU(inplace=True)
            ]
            curr_dim = curr_dim // 2

        # Output
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# PatchGAN discriminator
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super().__init__()

        # A 70x70 PatchGAN
        model = [
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  # output patch
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# ---------------------------
# Instantiate models
# ---------------------------
G_AB = GeneratorResNet(opt.channels, opt.channels, n_residual_blocks=9).to(device)  # A -> B (G_G)
G_BA = GeneratorResNet(opt.channels, opt.channels, n_residual_blocks=9).to(device)  # B -> A (G_F)
D_A = Discriminator(opt.channels).to(device)  # discriminate real A vs fake A
D_B = Discriminator(opt.channels).to(device)  # discriminate real B vs fake B

G_AB.apply(weights_init_normal)
G_BA.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)

# ---------------------------
# Losses & optimizers
# ---------------------------
criterion_GAN = nn.MSELoss().to(device)
criterion_cycle = nn.L1Loss().to(device)
criterion_identity = nn.L1Loss().to(device)

optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()),
                               lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Schedulers (linear decay after decay_epoch)
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, 0, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, 0, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, 0, opt.decay_epoch).step)

# Buffers for previously generated images (optional but helpful)
class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        returned = []
        for element in data.detach().cpu():
            element = element.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                returned.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    tmp = self.data[i].clone()
                    self.data[i] = element
                    returned.append(tmp)
                else:
                    returned.append(element)
        return torch.cat(returned).to(device)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# ---------------------------
# Helper functions
# ---------------------------
def sample_images(batches_done, real_A, real_B):
    """Saves a generated sample from the validation set"""
    G_AB.eval()
    G_BA.eval()
    with torch.no_grad():
        fake_B = G_AB(real_A)
        recov_A = G_BA(fake_B)
        fake_A = G_BA(real_B)
        recov_B = G_AB(fake_A)

    imgs = torch.cat([real_A, fake_B, recov_A, real_B, fake_A, recov_B], 0)
    imgs = (imgs + 1) / 2  # denorm to [0,1]
    os.makedirs("training_images", exist_ok=True)
    save_image(imgs, "training_images/%s.png" % batches_done, nrow=real_A.size(0))
    G_AB.train()
    G_BA.train()

def save_checkpoint(epoch):
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "G_AB": G_AB.state_dict(),
        "G_BA": G_BA.state_dict(),
        "D_A": D_A.state_dict(),
        "D_B": D_B.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D_A": optimizer_D_A.state_dict(),
        "optimizer_D_B": optimizer_D_B.state_dict(),
        "epoch": epoch
    }, f"checkpoints/cyclegan_epoch_{epoch}.pth")

# ---------------------------
# Training
# ---------------------------
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

prev_time = None
batches_done = 0

# ---------------------------
# Training loop
# ---------------------------
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

print("Starting training...")
batches_done = 0

for epoch in range(opt.n_epochs):
    loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{opt.n_epochs}")
    
    for i, batch in loop:
        real_A = batch["A"].to(device)
        real_B = batch["B"].to(device)

        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()

        # Identity loss
        same_A = G_BA(real_A)
        loss_id_A = criterion_identity(same_A, real_A) * opt.lambda_id

        same_B = G_AB(real_B)
        loss_id_B = criterion_identity(same_B, real_B) * opt.lambda_id

        # GAN loss
        fake_B = G_AB(real_A)
        pred_fake_B = D_B(fake_B)
        valid_B = torch.ones_like(pred_fake_B, requires_grad=False)
        loss_GAN_AB = criterion_GAN(pred_fake_B, valid_B)

        fake_A = G_BA(real_B)
        pred_fake_A = D_A(fake_A)
        valid_A = torch.ones_like(pred_fake_A, requires_grad=False)
        loss_GAN_BA = criterion_GAN(pred_fake_A, valid_A)

        # Cycle loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A) * opt.lambda_cyc

        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B) * opt.lambda_cyc

        # Total generator loss
        loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B + loss_id_A + loss_id_B
        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------
        optimizer_D_A.zero_grad()

        pred_real_A = D_A(real_A)
        valid_A = torch.ones_like(pred_real_A, requires_grad=False)
        loss_D_real_A = criterion_GAN(pred_real_A, valid_A)

        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        pred_fake_A = D_A(fake_A_.detach())
        fake_A_tensor = torch.zeros_like(pred_fake_A, requires_grad=False)
        loss_D_fake_A = criterion_GAN(pred_fake_A, fake_A_tensor)

        loss_D_A_val = 0.5 * (loss_D_real_A + loss_D_fake_A)
        loss_D_A_val.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------
        optimizer_D_B.zero_grad()

        pred_real_B = D_B(real_B)
        valid_B = torch.ones_like(pred_real_B, requires_grad=False)
        loss_D_real_B = criterion_GAN(pred_real_B, valid_B)

        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        pred_fake_B = D_B(fake_B_.detach())
        fake_B_tensor = torch.zeros_like(pred_fake_B, requires_grad=False)
        loss_D_fake_B = criterion_GAN(pred_fake_B, fake_B_tensor)

        loss_D_B_val = 0.5 * (loss_D_real_B + loss_D_fake_B)
        loss_D_B_val.backward()
        optimizer_D_B.step()
        
        # --------------
        #  Log Progress
        # --------------
        batches_done += 1
        loop.set_postfix({
            "loss_G": f"{loss_G.item():.4f}",
            "loss_D_A": f"{loss_D_A_val.item():.4f}",
            "loss_D_B": f"{loss_D_B_val.item():.4f}"
        })

        # Save sample images periodically
        if batches_done % opt.save_interval == 0:
            # use a small validation batch to visualize
            try:
                val_batch = next(iter(val_dataloader))
                sample_images(batches_done, val_batch["A"].to(device), val_batch["B"].to(device))
            except Exception as e:
                # no test data or other issue, fallback to saving current results
                sample_images(batches_done, real_A, real_B)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save checkpoint every epoch (or change as you like)
    save_checkpoint(epoch + 1)
    print(f"Epoch {epoch+1}/{opt.n_epochs} checkpoint saved.")

print("Training finished.")
