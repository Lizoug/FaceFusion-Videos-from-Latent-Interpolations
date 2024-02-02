import gdown
import os

# The ID of the file on Google Drive
file_id = '1J4EUE5xSUsuS7cuaSqsTwyzW6ZxmAx5y'
# The name of the file to be saved
model_name = 'generator_model_batch_3462000.pt'
# The name of the new folder where the model will be saved
new_folder_name = 'ModelFolder'

# Create the new folder in the current working directory if it does not exist
if not os.path.exists(new_folder_name):
    os.makedirs(new_folder_name)

# The path where the model will be saved
destination = os.path.join(new_folder_name, model_name)
# The full URL to download the model
url = f'https://drive.google.com/uc?id={file_id}'

# Download the model and save it in the new folder
gdown.download(url, destination, quiet=False)