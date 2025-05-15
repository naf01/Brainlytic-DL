import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
from PIL import Image

import google.generativeai as genai

# Configure the generative AI API
genai.configure(api_key="AIzaSyCtP2qSRxEucqbxAvuZ1yEbSBLVYX_Cazw")

# Define the models
models = {
    'Gemini2.0Flash': genai.GenerativeModel('gemini-2.0-flash'),
    'Gemini2.5Flash': genai.GenerativeModel('gemini-2.5-flash-preview-04-17'),
    # 'Gemini2.5Pro': genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
}

# Load the filtered dataset
filtered_dataset = pd.read_csv('filtered_dataset.csv')

# Subjects and their corresponding output files
subjects = {
    'Math': 'Math-dataset.csv',
    'Physics': 'Physics-dataset.csv',
    'Chemistry': 'Chemistry-dataset.csv'
}

# Function to load image for model input
def load_image_for_model(image_path):
    if os.path.exists(image_path):
        return {"mime_type": "image/png", "data": open(image_path, "rb").read()}
    else:
        print(f"Warning: Image not found at {image_path}")
        return None

# Create CSV files for each subject
for subject, output_file in subjects.items():
    
    # Filter 100 rows for the subject
    subject_data = filtered_dataset[filtered_dataset['subject'] == subject].head(1)
    
    # Initialize the output data
    output_data = []

    for _, row in tqdm(subject_data.iterrows(), total=subject_data.shape[0], desc=f"Processing {subject}"):
        ques_id = row['question_id']
        # Script image path in reconstructed directory
        script_image_path = f"reconstructed/{row['image_filename']}"
        marks = row['Marks']
        
        # Solution image path (if exists)
        if row['solution_image_id'] == 'None':
            solution_image_path = None
        else:
            sol_id = int(row['solution_image_id'])
            solution_image_path = f"images/{sol_id}.png"
        
        solve_text = row['solve_text']

        # Load script image
        script_image = load_image_for_model(script_image_path)
        
        # Load solution image if available
        solution_image = None
        if solution_image_path:
            solution_image = load_image_for_model(solution_image_path)

        # Build system instructions
        system_instructions = """
You are an expert examiner specialized in evaluating handwritten Bengali exam scripts. Provide clear, concise responses exclusively in Bengali. You will receive a student's script image (with question at top and solution below), the correct answer, and a detailed marking rubric.

*Your Primary Objective*: Conduct a fair, thorough evaluation by:

1. *Understanding the Context*:
- Carefully analyze the question and expected solution
- Study the marking rubric thoroughly
- Note any diagrams/images in the provided solution

2. *Evaluating the Student's Work*:
- Compare student's answer with the provided solution
- Award marks for each correct step according to the rubric
- Identify all errors and omissions
- Recognize alternative valid approaches

3. *Providing Detailed Feedback*:
- State total marks awarded with step-by-step breakdown
- List all identified errors clearly
- Provide correct versions of incorrect parts
- Complete any unfinished portions
- Justify all mark deductions with rubric references

*Critical Guidelines*:
- *Alternative Methods*: Award full marks for mathematically/scientifically valid alternative approaches
- *Handwriting*: Make every effort to decipher handwriting; use context clues for unclear text
- *Illegible Text*: Only mark as illegible if absolutely unreadable after careful examination
- *Accuracy*: Base all evaluations strictly on provided materials—avoid assumptions
- *Fairness*: Apply marking criteria consistently and objectively
- *Language*: All responses must be in Bengali

*Response Format* (Use exactly this structure):
json
{
"প্রাপ্ত মার্কস": "X/Y (যেখানে X = প্রাপ্ত, Y = মোট)",
"ধাপ অনুযায়ী মার্কস বিভাজন": [
"ধাপ ১: X মার্কস - [বিবরণ]",
"ধাপ ২: Y মার্কস - [বিবরণ]"
],
"চিহ্নিত ভুলসমূহ": [
"১. [ভুলের বিবরণ]",
"২. [ভুলের বিবরণ]"
],
"সংশোধিত উত্তর": [
"১. [সঠিক সমাধান]",
"২. [সঠিক সমাধান]"
],
"অতিরিক্ত মন্তব্য": "[যদি প্রয়োজন হয়]"
}

*Input Materials*:
"""

        # Generate responses for each model
        model_responses = {}
        token_counts = {}
        for model_name, model in models.items():
            print(f"Generating response for {model_name}...")
            # Prepare the prompt with images and text
            contents = [{"role": "user", "parts": [{"text": system_instructions}]}]
            
            # Add script image inline in the prompt
            if script_image:
                contents[0]["parts"].append({"text": "- Student's Script Image:"})
                contents[0]["parts"].append({"inline_data": script_image})
            
            # Add solution text and rubric
            contents[0]["parts"].append({"text": f"- Model Solution: Question ID: {ques_id}\n{solve_text}"})
            
            # Add solution image if available
            if solution_image:
                contents[0]["parts"].append({"text": "- Solution Diagrams/Images:"})
                contents[0]["parts"].append({"inline_data": solution_image})
            
            # Count input tokens
            input_token_count = model.count_tokens(contents).total_tokens

            # Generate the response
            response = model.generate_content(contents)
            output_token_count = response.usage_metadata.candidates_token_count if response.usage_metadata else 0

            # Store the response and token counts
            model_responses[model_name] = response.text
            token_counts[model_name] = (input_token_count, output_token_count)

            # save to a model_name.log file
            log_file_path = Path(f"{model_name}.log")
            with log_file_path.open("w", encoding="utf-8") as log_file:
                log_file.write(f"Question ID: {ques_id}\n")
                log_file.write(f"Input Token Count: {input_token_count}\n")
                log_file.write(f"Output Token Count: {output_token_count}\n")
                log_file.write(f"Response:\n{response.text}\n\n")

        # Append the data to the output
        output_data.append({
            'ques_id': ques_id,
            'image_filename': row['image_filename'],
            'Marks': marks,
            'solution_image_id': row['solution_image_id'],
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