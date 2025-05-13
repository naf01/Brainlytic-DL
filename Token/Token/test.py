import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
image = Image.open('1002.jpg').convert('RGB')
data = np.array(image)

# Improved red detection
red = data[:,:,0]
green = data[:,:,1]
blue = data[:,:,2]
red_mask = (red > 100) & (red > green + 30) & (red > blue + 30)

# Create a copy to modify
output_data = data.copy()
output_data2 = data.copy()
# Replace red pixels with white (or nearby gray)
output_data[red_mask] = [255, 255, 255]  # White


# Save and display
output_image = Image.fromarray(output_data)
output_image.save('image_without_red_marks.jpg')


black_mask = (red < 50) & (green < 50) & (blue < 50)
near_white_mask = (red > 220) & (green > 220) & (blue > 220)

# Valid candidates: near-white pixels that are not red or black
valid_candidates_mask = near_white_mask 
candidate_coords = np.argwhere(valid_candidates_mask)

# Extract red pixel coordinates
red_coords = np.argwhere(red_mask)

# Create a copy for modification
output_data2 = data.copy()

# Replace red pixels with randomly selected near-white pixels
for y, x in red_coords:
    cy, cx = candidate_coords[np.random.randint(len(candidate_coords))]
    output_data2[y, x] = data[cy, cx]


output_image2 = Image.fromarray(output_data2)
output_image2.save('image_red_replaced_with_random_white.jpg')

output_data3 = data.copy()

valid_candidates_mask = near_white_mask & ~red_mask & ~black_mask
candidate_coords = np.argwhere(valid_candidates_mask)
for y, x in red_coords:
    cy, cx = candidate_coords[np.random.randint(len(candidate_coords))]
    output_data3[y, x] = data[cy, cx]

output_image3 = Image.fromarray(output_data3)
output_image3.save('image_red_replaced_with_random_white_in_silver_background.jpg')


fig, axs = plt.subplots(2, 2, figsize=(12, 6))  # create 2 subplots

# First image: original
axs[0,0].imshow(image)
axs[0,0].axis('off')
axs[0,0].set_title('Original Image')

# Second image: red removed
axs[0,1].imshow(output_data)
axs[0,1].axis('off')
axs[0,1].set_title('Image with Red Marks Removed')

axs[1,0].imshow(output_data2)
axs[1,0].axis('off')
axs[1,0].set_title('Image with white replaced randomly')

axs[1,1].imshow(output_data3)
axs[1,1].axis('off')
axs[1,1].set_title('Image with white replaced randomly in silver')



plt.tight_layout()
plt.show()

