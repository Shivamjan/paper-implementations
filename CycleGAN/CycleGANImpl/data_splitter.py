#!/usr/bin/env python3
"""
Prepare Celeb -> Anime dataset for CycleGAN
- Downsamples / balances both domains
- Resizes to 256x256
- Shuffles and splits into train/test sets
- Optionally augments the smaller domain (Anime)
- Creates folder structure: trainA (celebs), trainB (anime), testA, testB
"""

import os
import random
import shutil
from PIL import Image, ImageEnhance, ImageOps

# ---------------------------
# User settings
# ---------------------------
CELEB_DIR = "../data/celeba/img_align_celeba/all/"  
ANIME_DIR = "./images/"           
OUTPUT_DIR = "./data/celebs2anime"

IMG_SIZE = 128              
TEST_SPLIT = 0.1               
SEED = 42
AUGMENT_ANIME = True           # Apply flips/rotations/color jitter to anime images

# ---------------------------
# Reproducibility
# ---------------------------
random.seed(SEED)

# ---------------------------
# Helper functions
# ---------------------------
import os
import shutil

def reset_output_dir():
    if os.path.exists(OUTPUT_DIR):
        for root, dirs, files in os.walk(OUTPUT_DIR, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                try:
                    os.remove(file_path)
                except FileNotFoundError:
                    print(f"Skipped missing file: {file_path}")
            for name in dirs:
                dir_path = os.path.join(root, name)
                try:
                    os.rmdir(dir_path)
                except FileNotFoundError:
                    print(f"Skipped missing dir: {dir_path}")
        try:
            os.rmdir(OUTPUT_DIR)
        except FileNotFoundError:
            print(f"Skipped missing OUTPUT_DIR: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"{OUTPUT_DIR} reset successfully.")


def save_image(img, dst_path):
    """Resize and save image"""
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    img.save(dst_path, "JPEG", quality=95)

def copy_and_resize(src_files, dst_dir):
    """Copy and resize images"""
    os.makedirs(dst_dir, exist_ok=True)
    for f in src_files:
        try:
            img = Image.open(f).convert("RGB")
            base_name = os.path.basename(f)
            save_image(img, os.path.join(dst_dir, base_name))
        except Exception as e:
            print(f"Skipping {f}: {e}")

def augment_image(img):
    """Random augment: flip, rotation, brightness, color"""
    if random.random() > 0.5:
        img = ImageOps.mirror(img)
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, expand=True)
    w, h = img.size
    min_side = min(w, h)
    left = (w - min_side) // 2
    top = (h - min_side) // 2
    img = img.crop((left, top, left + min_side, top + min_side))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))
    return img

def copy_and_augment(src_files, dst_dir, target_count):
    """Copy original images and augment until reaching target_count"""
    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    n_src = len(src_files)
    while count < target_count:
        f = random.choice(src_files)
        try:
            img = Image.open(f).convert("RGB")
            # Save original
            save_image(img, os.path.join(dst_dir, f"orig_{count}.jpg"))
            count += 1
            # Save augmented version 
            if AUGMENT_ANIME and count < target_count:
                aug_img = augment_image(img)
                save_image(aug_img, os.path.join(dst_dir, f"aug_{count}.jpg"))
                count += 1
        except Exception as e:
            print(f"Skipping {f}: {e}")

# ---------------------------
# Main script
# ---------------------------
reset_output_dir()

# --- Celeb images (Domain A) ---
celeb_files = [os.path.join(CELEB_DIR, f) for f in os.listdir(CELEB_DIR)
               if f.lower().endswith((".png", ".jpg", ".jpeg"))]
random.shuffle(celeb_files)

# Downsample Celebs to match Anime size (~63k)
CELEB_TARGET = 63000
celeb_files = celeb_files[:CELEB_TARGET]

n_test = int(len(celeb_files) * TEST_SPLIT)
celeb_test, celeb_train = celeb_files[:n_test], celeb_files[n_test:]

copy_and_resize(celeb_train, os.path.join(OUTPUT_DIR, "trainA"))
copy_and_resize(celeb_test, os.path.join(OUTPUT_DIR, "testA"))

print(f"Celeb images: {len(celeb_train)} train, {len(celeb_test)} test")

# --- Anime images (Domain B) ---
anime_files = [os.path.join(ANIME_DIR, f) for f in os.listdir(ANIME_DIR)
               if f.lower().endswith((".png", ".jpg", ".jpeg"))]
random.shuffle(anime_files)

ANIME_TARGET = 63000
n_test = int(len(anime_files) * TEST_SPLIT)
anime_test, anime_train = anime_files[:n_test], anime_files[n_test:]

# Copy test images
copy_and_resize(anime_test, os.path.join(OUTPUT_DIR, "testB"))

# Copy + augment train images to reach target count
copy_and_augment(anime_train, os.path.join(OUTPUT_DIR, "trainB"), ANIME_TARGET)

print(f"Anime images: {ANIME_TARGET} train, {len(anime_test)} test")
print("Dataset preparation completed!")
