import os
import glob
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


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

def convert_img_to_tensor(img_path, applied_transforms):
    image = Image.open(img_path)
    tensor_image = applied_transforms(image)
    return tensor_image


if __name__ == "__main__":
    IMG_SIZE = 64
    DATASET_DIR = r"C:\Users\lizak\Data_Science\Semester_5\Advanced_IS\Project_Data\img_align_celeba\img_align_celeba"

    image_paths = get_image_path(DATASET_DIR)
    transform_pipeline = transform(IMG_SIZE)

    # Convert all images in the dataset to tensors
    dataset = [convert_img_to_tensor(img_path, transform_pipeline) for img_path in image_paths]

    # Dataloader creation
    BATCH_SIZE = 32
    data_loader = DataLoader(dataset=dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True)

    print(f"Total batches in dataloader: {len(data_loader)} with batch size of {BATCH_SIZE}")

    # Display sample images
    sample_imgs = next(iter(data_loader))

    # Check the shape
    print(sample_imgs.shape)

    # Plotting
    figure, axes = plt.subplots(3, 5, figsize=(14, 9))
    for idx, ax in enumerate(axes.flat):
        image_array = sample_imgs[idx].numpy()
        image_array = np.transpose(image_array, (1, 2, 0))
        ax.imshow(image_array)
        ax.axis('off')
    plt.show()