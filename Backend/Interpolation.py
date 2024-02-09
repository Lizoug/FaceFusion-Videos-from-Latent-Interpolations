import numpy as np
import torch
import cv2
import os
from PIL import Image


def generate_latent_vector(seed, size):
    np.random.seed(seed)
    return torch.randn(size).float()  # Generate the vector


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

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def create_gif(image_folder, gif_name, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frames = [Image.open(os.path.join(image_folder, image)) for image in sorted(images)]

    frame_duration = int(1000 / fps)  # Duration of each frame in milliseconds
    frames[0].save(
        gif_name,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=frame_duration,
        loop=0)


def main(main, steps, gif_fps):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the entire pretrained Generator model
    generator = torch.load(r"./../../../Model_128x128_end/generator_model_batch_3924000.pt")
    generator.to(device).eval()

    # Base output directory
    base_output_dir = "interpolation_videos"
    os.makedirs(base_output_dir, exist_ok=True)

    # video_folder = os.path.join(base_output_dir, "videos")
    # os.makedirs(video_folder, exist_ok=True)

    # For creating the GIF
    gif_folder = os.path.join(base_output_dir, "gifs")
    os.makedirs(gif_folder, exist_ok=True)

    for seed_pair in seeds:
        seed1, seed2 = seed_pair
        # Generate latent vectors and move them to the same device as the model
        # z1 = generate_latent_vector(seed1, 100).to(device)
        # z2 = generate_latent_vector(seed2, 100).to(device)
        z1 = generate_latent_vector(seed1, (1, 100)).to(device)
        z2 = generate_latent_vector(seed2, (1, 100)).to(device)

        # Image generation
        image_folder = os.path.join(
            base_output_dir,
            f"images_seed_{seed1}_to_{seed2}"
            )
        os.makedirs(image_folder, exist_ok=True)

        for i in range(steps):
            val = i / (steps - 1)
            z = lerp(val, z1, z2).unsqueeze(0)  # The function unsqueeze(0)
            # adds an extra dimension to make the vector compatible with the
            # expected input format of the generator.
            # Ensures that PyTorch does not compute gradients for the
            # following operations. This is useful for saving memory and
            # speeding up computations when only making predictions
            # (here: generating an image) and not updating the model
            with torch.no_grad():
                generated_image = generator(z)
            # squeeze(0) removes the extra dimension.
            # detach() separates the image from the current computation graph
            # to save memory. transpose(1, 2, 0) changes the dimension order
            # to convert the image format from PyTorch's
            # (Channel, Height, Width) to the format expected by OpenCV
            # (Height, Width, Channel).
            image = generated_image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
            # scales the pixel values, which were originally between -1 and 1,
            # to the range of 0 to 255.
            image = ((image + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR
            cv2.imwrite(os.path.join(image_folder, f"image_{i:04d}.png"), image)

       # video_name = os.path.join(video_folder, f"latent_space_exploration_seed_{seed1}_to_{seed2}.mp4")
       # create_video(image_folder, video_name, video_fps)

        gif_name = os.path.join(
            gif_folder,
            f"latent_space_exploration_seed_{seed1}_to_{seed2}.gif")
        create_gif(image_folder, gif_name, gif_fps)


if __name__ == "__main__":
    # seeds = [(1, 2), (60, 80), (333, 666)]  #  all the seed pairs
    seeds = [(np.random.randint(0, 500), np.random.randint(0, 1000)) for _ in range(50)]

    steps = 50  # Fewer steps to make the process faster
    # video_fps = 60  # Frame rate for video
    gif_fps = 200    # Frame rate for GIF; adjust this to change GIF speed

    main(seeds, steps, gif_fps)
