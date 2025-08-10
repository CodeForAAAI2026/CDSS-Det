import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# Set the paths and the split proportion
original_dir = ''
dir1 = ''
dir2 = ''
proportion_labeled = 0.18

# Create the new directories if they don't exist
os.makedirs(dir1, exist_ok=True)
os.makedirs(dir2, exist_ok=True)

# Get all case directories
case_dirs = [d for d in os.listdir(original_dir) if os.path.isdir(os.path.join(original_dir, d))]

# Split case directories into labeled and unlabeled
case_dirs_labeled, case_dirs_unlabeled = train_test_split(case_dirs, test_size=(1 - proportion_labeled), random_state=42)

# Copy labeled data to dir1
for case_dir in case_dirs_labeled:
    src_dir = os.path.join(original_dir, case_dir)
    dst_dir = os.path.join(dir1, case_dir)
    shutil.copytree(src_dir, dst_dir)

# Copy unlabeled data to dir2 and remove label.npy
for case_dir in case_dirs_unlabeled:
    src_dir = os.path.join(original_dir, case_dir)
    dst_dir = os.path.join(dir2, case_dir)
    shutil.copytree(src_dir, dst_dir)
    label_path = os.path.join(dst_dir, 'label.npy')
    if os.path.exists(label_path):
        os.remove(label_path)

print(f"Data split completed. Labeled data in '{dir1}' and unlabeled data in '{dir2}'.")
