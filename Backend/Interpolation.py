import numpy as np
import torch
from Generator import Generator
import numpy as np
import cv2
import os
import argparse

def generate_latent_vector(seed, size):
    np.random.seed(seed)
    return torch.from_numpy(np.random.normal(0, 1, size)).float()

def lerp(val, v1, v2):
    """Linear interpolation between low and high with weighting val."""
    return (1.0 - val) * v1 + val * v2

def slerp(val, v1, v2):
    # Normalize the vectors
    norm_v1 = torch.norm(v1)
    norm_v2 = torch.norm(v2)

    # Compute the dot product of the normalized vectors
    # Wenn v1 und v2 Einheitsvektoren sind, ist das Skalarprodukt 
    # direkt der Cosinus des Winkels zwischen ihnen.
    dot_product = torch.dot(v1 / norm_v1, v2 / norm_v2)

    # Ensure that the dot product stays within the valid range for arccos
    clipped_dot_product = np.clip(dot_product, -1, 1)

    # Calculate the angle omega using arccos
    omega = np.arccos(clipped_dot_product)

    # Calculate the sine of Omega
    sin_omega = np.sin(omega)

    # Compute the scaling factors for the interpolation
    scale_v1 = np.sin((1.0 - val) * omega) / sin_omega
    scale_v2 = np.sin(val * omega) / sin_omega

    # Perform the interpolation and return the result
    return scale_v1 * v1 + scale_v2 * v2

def create_video(image_folder, video_name, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Try using a different codec here
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def main():
    # Hardcoded seeds
    seed1 = 123  
    seed2 = 456 
    steps = 3000  
    fps = 30     # Example Frames per Second for the video

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the entire pretrained Generator model
    generator = torch.load(r"C:\Users\lizak\Data_Science\Semester_5\Advanced_IS\Project\FaceFusion-Videos-from-Latent-Interpolations\Backend\Model\generator_model_batch_636000.pt")
    generator.to(device).eval()

    # Generate latent vectors and move them to the same device as the model
    z1 = generate_latent_vector(seed1, 100).to(device)
    z2 = generate_latent_vector(seed2, 100).to(device)

    # Image generation
    image_folder = "generated_images"
    os.makedirs(image_folder, exist_ok=True)

    for i in range(steps):
        val = i / (steps - 1)
        z = lerp(val, z1, z2).unsqueeze(0) # The function unsqueeze(0) adds an extra dimension to make the vector compatible with the expected input format of the generator.
        # Ensures that PyTorch does not compute gradients for the following operations. 
        # This is useful for saving memory and speeding up computations when only making 
        # predictions (here: generating an image) and not updating the model
        with torch.no_grad():
            generated_image = generator(z)
        # squeeze(0) removes the extra dimension.
        # detach() separates the image from the current computation graph to save memory.
        # transpose(1, 2, 0) changes the dimension order to convert the image format from PyTorch's 
        # (Channel, Height, Width) to the format expected by OpenCV (Height, Width, Channel).
        image = generated_image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        # scales the pixel values, which were originally between -1 and 1, to the range of 0 to 255.
        image = ((image + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(image_folder, f"image_{i:04d}.png"), image)

    # Create the video
    create_video(image_folder, "latent_space_exploration.mp4", fps)

if __name__ == "__main__":
    main()
