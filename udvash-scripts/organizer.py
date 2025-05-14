import csv

# Define the ranges to exclude
exclude_ranges = [
    (1091, 1321),
    (1620, 1892),
    (2156, 2274),
    (2436, 2657),
    (2894, 2982),
    (3203, 3466),
    (4036, 4201),
    (3674, 3832),
    (4304, 4327)
]

# Function to check if a number falls in any of the exclude ranges
def is_excluded(image_number):
    for start, end in exclude_ranges:
        if start <= image_number <= end:
            return True
    return False

# Read from the input CSV and filter the data
filtered_data = []
subject_counts = {}
with open('evaluation_results.csv', 'r', newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        # Extract the image number from the filename
        image_number = int(row['image_filename'].split('.')[0])
        # Check if the image number is in the exclude ranges
        if not is_excluded(image_number):
            filtered_data.append(row)
            # Count the subjects
            subject = row['subject']
            if subject in subject_counts:
                subject_counts[subject] += 1
            else:
                subject_counts[subject] = 1

# Sort the filtered data by subject (Math, Physics, Chemistry)
filtered_data.sort(key=lambda x: x['subject'])

# Write the filtered and sorted data to a new CSV file
with open('filtered_dataset.csv', 'w', newline='', encoding='utf-8') as outfile:
    fieldnames = ['question_id', 'image_filename', 'Marks', 'solution_image_id', 'solve_text', 'subject']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(filtered_data)

# Print the counts for each subject
print("Filtered and sorted dataset has been saved to 'filtered_dataset.csv'.")
print("Counts for each subject:")
for subject, count in subject_counts.items():
    print(f"{subject}: {count}")