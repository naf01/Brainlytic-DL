import os
import pandas as pd

folders = ['Physics', 'Chemistry', 'Math']

image_filenames = set()

for folder in folders:
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') and not filename.endswith('_reconstructed.jpg'):
            image_filenames.add(filename)


csv_file = 'Udvash Data - Final.csv'
df = pd.read_csv(csv_file)


filtered_df = df[df['image_filename'].isin(image_filenames)]

filtered_df.to_csv('Filtered_Udvash_Data.csv', index=False)

print(f"Filtered {len(filtered_df)} rows. Saved to 'Filtered_Udvash_Data.csv'.")
