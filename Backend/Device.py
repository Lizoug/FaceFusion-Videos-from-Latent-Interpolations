import torch

# checking if im using CUDA
# We need to determine the best device for us
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE is", DEVICE)