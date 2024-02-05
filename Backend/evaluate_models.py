import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Define a list of seeds
# seeds = [23, 59, 125, 200, 98, 126, 333, 100]
# seeds = [124, 120, 60, 620, 999, 541, 26]
seeds = [120, 620]
# Define directories
current_dir = os.path.dirname(os.path.realpath(__file__))

# Where the models are stored
model_dir = os.path.abspath(os.path.join(current_dir, "../../../Model_128x128_end"))
# Where the generated images should be saved
output_dir_base = os.path.join("..", "Generated_Images")
print("Absolute model directory:", os.path.abspath(model_dir))
print("Absolute output directory base:", os.path.abspath(output_dir_base))


# Size of the latent vector
nz = 100

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to save images
def save_image(image, path, filename):
    image = image.detach().cpu().numpy().transpose(1, 2, 0)
    image = (image + 1) / 2
    plt.imsave(os.path.join(path, filename), image.clip(0, 1))


# For each seed
for seed in seeds:
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate a consistent random latent vector
    fixed_vector = torch.randn(1, nz, device=device)

    # Create output directory for this seed
    output_dir = os.path.join(output_dir_base, f"seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)

    # For each model in the directory
    for file in os.listdir(model_dir):
        if file.startswith("generator") and file.endswith(".pt"):
            # Extract batch number
            batch_number = re.search(r"batch_(\d+)", file).group(1)
            model_path = os.path.join(model_dir, file)
            model = torch.load(model_path)
            model = model.to(device)
            model.eval()

            # Generate image
            with torch.no_grad():
                generated_image = model(fixed_vector)

            # Save the generated image with batch number in the filename
            save_image(
                generated_image.squeeze(),
                output_dir,
                f"generator_batch_{batch_number}.png")
