import os
import random
from pathlib import Path

# Set the path to the Kvasir dataset
kvasir_dataset_path = Path('datasets/kvasir/kvasir-dataset-v2/JPEGImages/')

# Set the split ratio for train and validation sets
train_ratio = 0.8

# Define the classes
classes = ['dyed-resection-margins', 'esophagitis', 'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps']

# Create lists to store image paths for each set
train_images = []
val_images = []
trnval_images = []

# Iterate through each class
for class_name in classes:
    class_path = kvasir_dataset_path / 'kvasir-dataset-v2' / class_name / 'images'
    image_paths = [str(p) for p in class_path.glob('*.*')]
    random.shuffle(image_paths)

    # Split the images into train and validation sets
    split_idx = int(len(image_paths) * train_ratio)
    train_images.extend([f'{os.path.basename(p)} {class_name}' for p in image_paths[:split_idx]])
    val_images.extend([f'{os.path.basename(p)} {class_name}' for p in image_paths[split_idx:]])
    trnval_images.extend([f'{os.path.basename(p)} {class_name}' for p in image_paths])

# Write the image IDs to text files
with open('trn.txt', 'w') as f:
    f.write('\n'.join(train_images))

with open('val.txt', 'w') as f:
    f.write('\n'.join(val_images))

with open('trnval.txt', 'w') as f:
    f.write('\n'.join(trnval_images))