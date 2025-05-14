import pandas as pd
from pathlib import Path
from tqdm import tqdm

import google.generativeai as genai

# Configure the generative AI API
genai.configure(api_key="AIzaSyBSISlUTjixUxp6_eIAHnMEIdVmTHnlyhE")

# Define the system instructions
SYSTEM_INSTRUCTIONS = """
You are a script checker who excels in evaluating and reviewing exam scripts in Bengali. Your response should be in Bengali too. You will be provided the image of a student’s handwritten script, the answer to the question along with the marking rubric. At the top of the image is the problem statement, followed by the student’s written solution. 

Develop your understanding of the problem statement and the image context first and relate to the solution properly. The solution may contain images which can aid your evaluation process by providing necessary information.
Your job is to evaluate the script fairly, awarding a score out of 10 according to the rubric.
The student may follow steps which are different from the provided solution, and can still be correct, which is good. You should consider this while evaluating the script. Make sure you correctly grasp hand writing no matter how worse it is. Extrapolate from the surrounding if any letter or word is incomplete.
 
Pinpoint any mistakes clearly and explain why they are incorrect. Do not repeat the full corrected steps; only highlight what was wrong. Provide concise, constructive remarks for the student at the end.

You will be provided:
Script Image
Solution and Rubric
Solution Images (If any): {Solution}


Response Structure:

মার্কস: (total marks, and then showing how much mark is given for which correct step)
চিহ্নিত ভুল: (bulleted and concise)
ভুল গুলোর সঠিক: (bulleted and concise)
অসম্পূর্ণ উত্তরের ক্ষেত্রে বাকি অংশের সমাধান: (concise)
মার্কস কাটার কারণ: (bulleted and concise, showing how much mark is deducted for which mistake)
"""

# Define the models
models = {
    'Gemini2.0Flash': genai.GenerativeModel('gemini-2.0-flash'),
    'Gemini2.5Flash': genai.GenerativeModel('gemini-2.5-flash-preview-04-17'),
    'Gemini2.5Pro': genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
}

# Load the filtered dataset
filtered_dataset = pd.read_csv('filtered_dataset.csv')

# Subjects and their corresponding output files
subjects = {
    'Math': 'Math-dataset.csv',
    'Physics': 'Physics-dataset.csv',
    'Chemistry': 'Chemistry-dataset.csv'
}

# Create CSV files for each subject
for subject, output_file in subjects.items():
    # Filter 100 rows for the subject
    subject_data = filtered_dataset[filtered_dataset['subject'] == subject].head(1)
    
    # Initialize the output data
    output_data = []

    for _, row in tqdm(subject_data.iterrows(), total=subject_data.shape[0], desc=f"Processing {subject}"):
        ques_id = row['question_id']
        image_filename = f"images/{row['image_filename']}"
        marks = row['Marks']
        if row['solution_image_id'] == 'None':
            solution_image_id = None
        else:
            solution_image_id = f"images/{row['solution_image_id']}.png"
        solve_text = row['solve_text']

        # Generate responses for each model
        model_responses = {}
        token_counts = {}
        for model_name, model in models.items():
            # Prepare the prompt
            contents = [SYSTEM_INSTRUCTIONS, f"Question ID: {ques_id}", solve_text]
            input_token_count = model.count_tokens(contents).total_tokens

            # Generate the response
            response = model.generate_content(contents)
            output_token_count = response.usage_metadata.candidates_token_count if response.usage_metadata else 0

            # Store the response and token counts
            model_responses[model_name] = response.text
            token_counts[model_name] = (input_token_count, output_token_count)

        # Append the data to the output
        output_data.append({
            'ques_id': ques_id,
            'image_filename': image_filename,
            'Marks': marks,
            'solution_image_id': solution_image_id,
            'solve_text': solve_text,
            'input_token_count': token_counts['Gemini2.0Flash'][0],
            'Gemini2.0 response': model_responses['Gemini2.0Flash'],
            'Gemini 2.0 output token count': token_counts['Gemini2.0Flash'][1],
            'Gemini2.5Flash response': model_responses['Gemini2.5Flash'],
            'Gemini 2.5 Flash output token count': token_counts['Gemini2.5Flash'][1],
            'Gemini2.5Pro response': model_responses['Gemini2.5Pro'],
            'Gemini 2.5 Pro output token count': token_counts['Gemini2.5Pro'][1]
        })

    # Save the output data to a CSV file
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_file, index=False)
    print(f"Saved {subject} dataset to {output_file}")