import os
from PIL import Image

def find_images_with_different_size(root_dir, width=640, height=960):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                file_path = os.path.join(root, file)
                with Image.open(file_path) as img:
                    if img.size != (width, height):
                        print(file_path)

if __name__ == "__main__":
    directory = "./train"
    find_images_with_different_size(directory)
