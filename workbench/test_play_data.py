import json
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from workbench.model import UNet

# Load configuration from config.json
try:
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    print("Error: config.json file not found.")
    exit(1)
except json.JSONDecodeError:
    print("Error: Failed to decode config.json.")
    exit(1)

# Extract directories and paths from the config
test_these_dir = config['directories']['retest_these']
output_dir = config['directories']['output']
model_save_path = config['paths']['model']['save_path']

# Extract image dimensions from the config
img_height = 960
img_width = 640
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = UNet()
# Ensure the model is loaded to CPU if CUDA is not available
try:
    model.load_state_dict(torch.load(model_save_path, map_location=device))
except FileNotFoundError:
    print(f"Error: Model file {model_save_path} not found.")
    exit(1)
except RuntimeError as e:
    print(f"Error loading the model: {e}")
    exit(1)

model = model.to(device)
model.eval()

# Define the transformations to apply to the input images
noisy_transform = transforms.Compose([
    transforms.Resize((img_height, img_width), antialias=True),            
    transforms.ToTensor(),
])

def process_single_image(input_img_path, output_img_path, noisy_transform):
    """
    Process a single image by applying transformations and model inference.

    Parameters:
    - input_img_path (str): Path to the input image.
    - output_img_path (str): Path where the processed image will be saved.
    - noisy_transform (torchvision.transforms.Compose): Transformations to apply to the input image.
    """
    try:
        input_img = Image.open(input_img_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Input image {input_img_path} not found.")
        return
    except IOError:
        print(f"Error: Failed to open {input_img_path}.")
        return

    # Apply the noisy_transform
    input_tensor = noisy_transform(input_img)
    
    # Add a batch dimension
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    # Run the model
    with torch.no_grad():
        output_batch = model(input_batch)
    
    # Remove the batch dimension and convert the tensor back to an image
    output_tensor = output_batch.squeeze(0).cpu()
    output_img = transforms.ToPILImage()(output_tensor)
    
    # Save the output image
    try:
        output_img.save(output_img_path)
    except IOError:
        print(f"Error: Failed to save {output_img_path}.")

def process_all_images(input_dir, output_dir, noisy_transform):
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        return
    """
    Process all images in a directory, applying transformations and saving the output.

    Parameters:
    - input_dir (str): Directory containing input images.
    - output_dir (str): Directory where processed images will be saved.
    - noisy_transform (torchvision.transforms.Compose): Transformations to apply to the input images.
    """

    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all files in the input directory
    for file_name in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, file_name)
        
        # Check if the file is an image
        if os.path.isfile(input_file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            output_file_path = os.path.join(output_dir, file_name).replace('_before', '_after')
            process_single_image(input_file_path, output_file_path, noisy_transform)

# Usage example with directories from config.json
process_all_images(test_these_dir, output_dir, noisy_transform)