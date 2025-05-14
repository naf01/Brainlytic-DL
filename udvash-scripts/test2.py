import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
import os

image_files = glob.glob('./images/*.jpg')  # Adjust the path to your images

for image_file in image_files:
    image = Image.open(image_file).convert('RGB')
    data = np.array(image)

    red = data[:, :, 0]
    green = data[:, :, 1]
    blue = data[:, :, 2]
    red_mask = (red > 100) & (red > green + 30) & (red > blue + 30)

    black_mask = (red < 50) & (green < 50) & (blue < 50)
    near_white_mask = (red > 220) & (green > 220) & (blue > 220)

    valid_candidates_mask = near_white_mask
    candidate_coords = np.argwhere(valid_candidates_mask)

    if len(candidate_coords) == 0:
        print(f"No near-white candidates found in {image_file}, skipping.")
        continue

    red_coords = np.argwhere(red_mask)
    output_data = data.copy()

    for y, x in red_coords:
        cy, cx = candidate_coords[np.random.randint(len(candidate_coords))]
        output_data[y, x] = data[cy, cx]

    output_image = Image.fromarray(output_data)
    base_name, ext = os.path.splitext(image_file)
    base_name = os.path.basename(base_name)
    output_filename = f"./reconstructed/{base_name}{ext}"
    output_image.save(output_filename)
    print(f"Processed {image_file} and saved to {output_filename}")

print("Processing complete.")