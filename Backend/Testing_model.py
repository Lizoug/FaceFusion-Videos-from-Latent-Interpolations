import torch
import matplotlib.pyplot as plt


# Load the pre-trained Generator model and set it to evaluation mode
generator = torch.load(r"c:\Users\lizak\Data_Science\Semester_5\Advanced_IS\Model_128x128_end\generator_model_batch_3918000.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = generator.to(device)
generator.eval()

# Generate random latent vectors
nz = 100  # Size of the latent vector
random_vectors = torch.randn(10, nz, device=device)  # Create 10 random vectors

# Generate images from the random vectors
with torch.no_grad():
    generated_images = generator(random_vectors)


# Function to display images
def show_images(images):
    fig, axes = plt.subplots(1, len(images), figsize=(20, 10))
    for img, ax in zip(images, axes):
        # Convert to numpy and change from (C, H, W) to (H, W, C)
        img = img.detach().cpu().numpy().transpose(1, 2, 0)
        # Scale to [0, 1] from [-1, 1]
        img = (img + 1) / 2
        ax.imshow(img.clip(0, 1))
        ax.axis('off')
    plt.show()


# Display the generated images
show_images(generated_images[:10])
