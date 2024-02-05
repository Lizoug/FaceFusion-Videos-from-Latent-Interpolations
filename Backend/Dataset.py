import os
import glob
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


# Define a class for the CelebA dataset that inherits from Dataset
class CelebADataset(Dataset):
    def __init__(self, dir, transform=None):
        # Get all image file paths from the given directory
        # find all files in the specified directory that match the given
        # pattern. In this case, it's looking for all .jpg files in the
        # specified directory. The function returns a list of file paths that
        # match the pattern.
        self.image_paths = sorted(glob.glob(os.path.join(dir, "*.jpg")))
        self.transform = transform  # Set the image transformation method

    def __len__(self):
        # Return the total number of images
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get an image by index and apply the transformation
        img_path = self.image_paths[idx]  # image path
        image = Image.open(img_path)  # open image
        if self.transform:
            image = self.transform(image)  # apply transformation
        return image


# Function to apply transformations to the images
def transform(size):
    return transforms.Compose([
        transforms.Resize(size=(size, size)),  # resizing the image
        # [0, 1]
        transforms.ToTensor(),  # Converting image to PyTorch tensor
        transforms.CenterCrop(size),  # Croping the image
        # [-1, 1]
        transforms.Normalize([0.5], [0.5])  # Normalize the image
    ])


if __name__ == "__main__":
    IMG_SIZE = 128  # Set the image size
    # Directory where the dataset is located
    current_script_dir = os.path.dirname(__file__)  # Gets the directory of the current script
    DATASET_DIR = os.path.join(current_script_dir, "../../../Project_Data/img_align_celeba/img_align_celeba")


    # Get the transformation pipelin
    transform_pipeline = transform(IMG_SIZE)

    # Create an instance of the dataset
    dataset = CelebADataset(DATASET_DIR, transform=transform_pipeline)
    BATCH_SIZE = 32  # batch size for data loading

    # DataLoader to load the dataset in batches
    data_loader = DataLoader(dataset=dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True)

    print(f"Total batches in dataloader: {len(data_loader)} with batch size of {BATCH_SIZE}")

    # Display sample images
    sample_imgs = next(iter(data_loader))

    figure, axes = plt.subplots(3, 5, figsize=(14, 9))  # subplot
    for idx, ax in enumerate(axes.flat):
        image_array = sample_imgs[idx].numpy()  # converting tensor to np array
        image_array = np.transpose(image_array, (1, 2, 0))  # Transpose array dimensions
        ax.imshow(image_array)  # Show image
        ax.axis('off')  # hide axis
    plt.show()  # display the plot
