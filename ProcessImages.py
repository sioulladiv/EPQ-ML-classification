import os
import pandas as pd
import cv2
from PIL import Image

# Parameters
image_dir = "Garbage classification"  # Directory containing the images
text_file = "zero-indexed-files.txt"  # Text file with image info
output_csv = "pixel_data_512x384.csv"  # Output CSV file
image_size = (128, 96)  # Resize all images to this size

with open(text_file, "r") as file:
    lines = file.readlines()

image_info = [line.strip().split() for line in lines]

rows = []

for image_filename, class_label in image_info:
    image_path = os.path.join(image_dir, image_filename)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        continue
    
    image = cv2.resize(image, image_size)
    flat_pixels = image.flatten().tolist()  
    flat_pixels.append(class_label)
    rows.append(flat_pixels)

columns = []
for y in range(image_size[1]):
    for x in range(image_size[0]):  
        columns.extend([f"R_{x}_{y}", f"G_{x}_{y}", f"B_{x}_{y}"])
columns.append("Class_Label")  

pixel_df = pd.DataFrame(rows, columns=columns)
pixel_df.to_csv(output_csv, index=False)

print(f"Pixel data saved to {output_csv}")
