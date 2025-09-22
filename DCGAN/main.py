import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms

import os
import numpy as np
import random
import pickle
import pandas as pd

# --- Seeding ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# --- Loss Tracker ---
class LossTracker:
    def __init__(self):
        self.g_losses = []
        self.d_losses = []

    def update(self, g_loss, d_loss):
        self.g_losses.append(g_loss)
        self.d_losses.append(d_loss)

    def get_recent_avg(self, window=100):
        if len(self.g_losses) < window:
            return np.mean(self.g_losses), np.mean(self.d_losses)
        return np.mean(self.g_losses[-window:]), np.mean(self.d_losses[-window:])

# --- Hyperparameters ---
img_size = 128
channels = 3
z_dim = 128

# --- Generator ---

'''
-> Input: Latent vector [batch_size, z_dim=128].
-> Linear layer: Maps to 512 * 8 * 8 = 32,768, reshaped to [batch_size, 512, 8, 8].
-> Four upsampling blocks with ConvTranspose2d, BatchNorm, and ReLU:
    8x8 → 16x16 → 32x32 → 64x64 → 128x128.
-> Output: [batch_size, 3, 128, 128] via ConvTranspose2d and Tanh (pixel range [-1, 1]).

~12–15M parameters, ~18 layers (9 learnable Conv/Linear).
'''
class Generator(nn.Module):
    def __init__(self, z_dim=128, img_size=128, channels=3):
        super(Generator, self).__init__()
        self.init_size = img_size // 16  # 128//16 = 8
        self.l1 = nn.Sequential(nn.Linear(z_dim, 512 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # 8 -> 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 16 -> 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),   # 32 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False), # 64 -> 128
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# --- Discriminator (Critic) ---
'''
-> In our discriminator, each Conv2d block with stride=2 halves the spatial resolution:
        128 → 64 → 32 → 16 → 8.
-> Instead of flattening and using a Linear layer, we apply a final Conv2d with kernel_size=8,
    which reduces [512, 8, 8] to a single scalar output [batch_size, 1].
'''

class Discriminator(nn.Module):
    def __init__(self, img_size=128, channels=3):
        super().__init__()

        def discriminator_block(in_filters, out_filters):
            layers = [
                nn.Conv2d(in_filters, out_filters, 4, 2, 1, bias=False),
                # in_channels = 3
                # out_channels = 64
                # kernel_size = 4x4
                # stride = 2   (halves the spatial dimensions → downsampling)
                # padding = 1  (keeps dimensions aligned for halving)
                nn.LeakyReLU(0.2, inplace=True)
            ]
            return layers

        final_size = img_size // 16

        self.model = nn.Sequential(
            *discriminator_block(channels, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=final_size, stride=1, padding=0, bias=False)
        )

    def forward(self, img):
        return self.model(img)
# --- Data ---
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
dataset = ImageFolder(root="./data/celeba/img_align_celeba/", transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

# --- Weight init ---
def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname or 'InstanceNorm' in classname:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# --- Models ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator(z_dim=z_dim, img_size=img_size, channels=channels).to(device)
D = Discriminator(img_size=img_size, channels=channels).to(device)
G.apply(weights_init)
D.apply(weights_init)

# --- Optimizers ---
optimizer_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizer_D = torch.optim.Adam(D.parameters(), lr=4e-4, betas=(0.5, 0.9)) 


# --- Gradient penalty ---
def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    d_interpolates = D(interpolates)
    grad_outputs = torch.ones_like(d_interpolates, device=device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = ((gradient_norm - 1) ** 2).mean()
    return gp 


# --- Training ---
lambda_gp = 5
n_critic = 5
fixed_noise = torch.randn(25, z_dim, device=device) 
loss_tracker = LossTracker()

no_epochs = 200
for epoch in range(no_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)
        cur_batch_size = imgs.size(0)

        # --- Train Discriminator ---
        optimizer_D.zero_grad()
        noise = torch.randn(cur_batch_size, z_dim, device=device)
        fake_imgs = G(noise).detach()
        real_validity = D(imgs)
        fake_validity = D(fake_imgs)
        
        gradient_penalty = compute_gradient_penalty(D, imgs, fake_imgs, device)
        
        D_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        D_loss.backward()
        optimizer_D.step()

        # --- Train Generator ---
        if i % n_critic == 0:
            optimizer_G.zero_grad()
            gen_imgs = G(noise) 
            G_loss = -torch.mean(D(gen_imgs))
            G_loss.backward()
            optimizer_G.step()

            loss_tracker.update(G_loss.item(), D_loss.item())

            if i % 200 == 0:
                print(f'[Epoch {epoch}/{no_epochs}] [Batch {i}/{len(dataloader)}] '
                      f'[D loss: {D_loss.item():.4f}] [G loss: {G_loss.item():.4f}]')

            batches_done = epoch * len(dataloader) + i
            if batches_done % 500 == 0:
                os.makedirs("images_wgan3", exist_ok=True)
                with torch.no_grad():
                    fake_samples = G(fixed_noise)
                    save_image(fake_samples,
                               f"images_wgan3/epoch_{epoch}_batch_{batches_done}.png",
                               nrow=5, normalize=True, value_range=(-1, 1))


# --- Save models and losses ---
torch.save(G.state_dict(), 'generator.pth')
torch.save(D.state_dict(), 'discriminator.pth')
with open('loss_tracker.pkl', 'wb') as f:
    pickle.dump(loss_tracker, f)
pd.DataFrame({'generator_loss': loss_tracker.g_losses,
              'discriminator_loss': loss_tracker.d_losses}).to_csv('training_losses.csv', index=False)

print("Training complete and assets saved.")
