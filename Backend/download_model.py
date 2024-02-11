import gdown
import os


# The file IDs and their original names in your Google Drive folder
files = {
    '1MbFbA4S_UecZPw4BhoaOc1H83QwluuZf': 'generator_model_batch_12000.pt',
    '17zuSetRxGMNBDQS1k7PJWLGM-15kyLap': 'generator_model_batch_300000.pt',
    '1t1Yy5uDD-zf-X-1XBSGi7aOdrHRwA7oC': 'generator_model_batch_2166000.pt',
    '1J4EUE5xSUsuS7cuaSqsTwyzW6ZxmAx5y': 'generator_model_batch_3462000.pt',
    '1j1xoJSWPUxM-GNpfxX0kDYNl1wYjR3Dz': 'generator_model_batch_3924000.pt'
}  # Replace the IDs and names with the actual ones

# The path to the directory two levels above the current directory
target_directory = '../../Model_128x128_end'  # This will be the common folder for all models

# Create the target directory if it does not exist
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# Download each file into the target directory
for file_id, original_name in files.items():
    # The full URL to download the file
    url = f'https://drive.google.com/uc?id={file_id}'

    # The file path where the file will be saved with its original name
    file_path = os.path.join(target_directory, original_name)

    # Download the file and save it in the target directory
    gdown.download(url, file_path, quiet=False, use_cookies=False)