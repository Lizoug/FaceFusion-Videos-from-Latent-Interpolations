import os
import glob
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

DATASET_DIR = "C:\Users\lizak\Data_Science\Semester_5\Advanced_IS\Project_Data\img_align_celeba\img_align_celeba"

def get_image_path(dir):
    return sorted(glob.glob(os.path.join(dir, "*.jpg")))


def transform(size):
    return transforms.Compose([
        transforms.Resize(size=(size, size)),
        # [0, 1]
        transforms.ToTensor(),
        transforms.CenterCrop(size),
        # [-1, 1]
        transforms.Normalize([0.5], [0.5])
    ])