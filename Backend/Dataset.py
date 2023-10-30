import os
import glob
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


class CelebADataset(Dataset):
    def __init__(self, dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(dir, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image
    

def transform(size):
    return transforms.Compose([
        transforms.Resize(size=(size, size)),
        # [0, 1]
        transforms.ToTensor(),
        transforms.CenterCrop(size),
        # [-1, 1]
        transforms.Normalize([0.5], [0.5])
    ])


if __name__ == "__main__":
    IMG_SIZE = 64
    DATASET_DIR = r"C:\Users\lizak\Data_Science\Semester_5\Advanced_IS\Project_Data\img_align_celeba\img_align_celeba"

    transform_pipeline = transform(IMG_SIZE)

    dataset = CelebADataset(DATASET_DIR, transform=transform_pipeline)
    BATCH_SIZE = 32
    data_loader = DataLoader(dataset=dataset, 
                             batch_size=BATCH_SIZE, 
                             shuffle=True)

    print(f"Total batches in dataloader: {len(data_loader)} with batch size of {BATCH_SIZE}")

    # Display sample images
    sample_imgs = next(iter(data_loader))

    figure, axes = plt.subplots(3, 5, figsize=(14, 9))
    for idx, ax in enumerate(axes.flat):
        image_array = sample_imgs[idx].numpy()
        image_array = np.transpose(image_array, (1, 2, 0))
        ax.imshow(image_array)
        ax.axis('off')
    plt.show()


