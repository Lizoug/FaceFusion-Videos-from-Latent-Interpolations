import torch
import numpy as np
from Interpolation import create_gif
import matplotlib.pyplot as plt
import os
import re
import cv2


# Function to generate images for a given list of seeds using pre-trained models
def generate_images(seeds, model_dir, output_dir_base, device, nz):
    # Iterate over each seed to generate images
    for seed in seeds:
        # Set the seed for PyTorch and NumPy to ensure reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create a fixed latent vector for the seed
        fixed_vector = torch.randn(1, nz, device=device)

        # Create an output directory specific to the current seed
        output_dir = os.path.join(output_dir_base, f"seed_{seed}")
        os.makedirs(output_dir, exist_ok=True)

        # Iterate over each model file in the model directory
        for file in os.listdir(model_dir):
            # Check if the file is a generator model file
            if file.startswith("generator") and file.endswith(".pt"):
                # Extract the batch number from the filename
                batch_number = re.search(r"batch_(\d+)", file).group(1)
                model_path = os.path.join(model_dir, file)

                # Load the model and move it to the specified device
                model = torch.load(model_path)
                model = model.to(device)
                model.eval()

                # Generate an image using the model without calculating gradients
                with torch.no_grad():
                    generated_image = model(fixed_vector)

                # Save the generated image using a predefined function
                save_image(generated_image.squeeze(), output_dir, f"generator_batch_{batch_number}.png")


def create_video(image_folder, video_name, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


# Function to save an image file
def save_image(image, path, filename):
    # Convert the tensor to a NumPy array and adjust its range to [0, 1]
    image = image.detach().cpu().numpy().transpose(1, 2, 0)
    image = (image + 1) / 2

    # Save the image to the specified path using matplotlib
    plt.imsave(os.path.join(path, filename), image.clip(0, 1))


if __name__ == "__main__":
    # Get the directory of the current script to build relative paths
    current_script_dir = os.path.dirname(__file__)

    # Define the model directory path relative to the current script
    model_dir = os.path.abspath(os.path.join(current_script_dir, "../../../Model_128x128_end"))

    # Define the base output directory for saving generated images
    output_dir_base = os.path.join(current_script_dir, "../All_Models_Generated_Images")

    # Select the computational device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Size of the latent vector used by the generator model
    nz = 100

    # Define the seeds to use for generating images
    seeds = [124, 120, 60, 620, 999, 541, 26]

    # Generate and save images for the defined seeds
    #generate_images(seeds, model_dir, output_dir_base, device, nz)

    # Specify the path to the folder containing images for a specific seed
    image_folder_path = os.path.join(output_dir_base, "seed_124")

    # Define the GIF directory and filename
    video_directory = os.path.join(current_script_dir, "../All_Models_Generated_Images/video")
    video_filename = "Model_eval.mp4"
    video_path = os.path.join(video_directory, video_filename)

    # Ensure the GIF directory exists
    os.makedirs(video_directory, exist_ok=True)

    
    # Call create_video function after generating images
    create_video(image_folder_path, video_path, 5)  # Adjust fps as needed
