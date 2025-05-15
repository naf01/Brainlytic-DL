import google.generativeai as genai
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import io

# Configure Gemini API
genai.configure(api_key="AIzaSyCtP2qSRxEucqbxAvuZ1yEbSBLVYX_Cazw")

# Define system instructions
SYSTEM_INSTRUCTIONS = """
You are an expert examiner specialized in evaluating handwritten Bengali exam scripts. Provide clear, concise responses exclusively in Bengali. You will receive a student's script image (with question at top and solution below), the correct answer, and a detailed marking rubric.

**Your Primary Objective**: Conduct a fair, thorough evaluation by:

1. **Understanding the Context**:
- Carefully analyze the question and expected solution
- Study the marking rubric thoroughly
- Note any diagrams/images in the provided solution

2. **Evaluating the Student's Work**:
- Compare student's answer with the provided solution
- Award marks for each correct step according to the rubric
- Identify all errors and omissions
- Recognize alternative valid approaches

3. **Providing Detailed Feedback**:
- State total marks awarded with step-by-step breakdown
- List all identified errors clearly
- Provide correct versions of incorrect parts
- Complete any unfinished portions
- Justify all mark deductions with rubric references

**Critical Guidelines**:
- **Alternative Methods**: Award full marks for mathematically/scientifically valid alternative approaches
- **Handwriting**: Make every effort to decipher handwriting; use context clues for unclear text
- **Illegible Text**: Only mark as illegible if absolutely unreadable after careful examination
- **Accuracy**: Base all evaluations strictly on provided materials—avoid assumptions
- **Fairness**: Apply marking criteria consistently and objectively
- **Language**: All responses must be in Bengali

**Response Format** (Use exactly this structure):
```json
{
    "প্রাপ্ত মার্কস": "X/Y (যেখানে X = প্রাপ্ত, Y = মোট(10))",
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
```

**Input Materials**:
"""

# Initialize the models
models = {
    'Gemini2.0Flash': genai.GenerativeModel('gemini-2.0-flash'),
    'Gemini2.5Flash': genai.GenerativeModel('gemini-2.5-flash-preview-04-17'),
    'Gemini2.5Pro': genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
}

# Load the original CSV data
df = pd.read_csv('Filtered_Udvash_Data_with_rubric.csv')

# Base directory for UDV structure
base_dir = Path('UDV')

# Subjects to process
subjects = ['Chemistry']

# Process each subject
for subject in subjects:
    print(f"Processing {subject}...")
    
    # Path to unevaluated images for this subject
    unevaluated_dir = base_dir / subject / 'Question Images' / 'Unevaluated Images'
    
    # Path to answer images for this subject
    answer_dir = base_dir / subject / 'Answer Images'
    
    # Skip if directory doesn't exist
    if not unevaluated_dir.exists():
        print(f"Directory {unevaluated_dir} does not exist, skipping...")
        continue
        
    # Get all unevaluated images
    unevaluated_images = list(unevaluated_dir.glob('*.jpg'))
    
    # Create a list to store results for this subject
    subject_results = []
    
    # Process each unevaluated image
    for img_path in tqdm(unevaluated_images, desc=f"Processing {subject} images"):
        # Get original image filename (without '_reconstructed' if present)
        if '_reconstructed' in img_path.stem:
            original_image_name = img_path.name.replace('_reconstructed', '')
        else:
            original_image_name = img_path.name
        
        # Find matching row in the dataframe
        row_match = df[df['image_filename'] == original_image_name]
        if row_match.empty:
            print(f"Image {original_image_name} not found in CSV, skipping...")
            continue
            
        row = row_match.iloc[0]
        
        # Get solution text and rubric
        solve_text = row.get('solve_text', '')
        rubric = row.get('rubric', '')
        solution_image_id = row.get('solution_image_id', None)
        
        # Prepare dictionary to store model outputs and token counts
        outputs = {}
        input_tokens = {}
        output_tokens = {}
        # print("Prompt stats")
        # print(f"Base Prompt Token Count: {models["Gemini2.0Flash"].count_tokens(SYSTEM_INSTRUCTIONS).total_tokens}")
        contents = [{"role": "user", "parts": [{"text": SYSTEM_INSTRUCTIONS}]}]
    
        # Add student image
        with Image.open(img_path) as script_img:
            contents[0]["parts"].append({"text": "- Student's Script Image:"})
            contents[0]["parts"].append({"inline_data": script_img})

        # print(f"After Adding Images Token Count: {models["Gemini2.0Flash"].count_tokens(contents).total_tokens}")
        contents[0]["parts"].append({"text": "- Solution:"})
        contents[0]["parts"].append({"text": solve_text})

        contents[0]["parts"].append({"text": "- Marking Rubric:"})
        contents[0]["parts"].append({"text": rubric})
        # print(f"After Adding Images, Solution and Rubric Token Count: {models["Gemini2.0Flash"].count_tokens(contents).total_tokens}")
        # Add answer image if available
        if pd.notna(solution_image_id):
            answer_image_id = f"{int(solution_image_id)}.png"
            answer_img_path = answer_dir / answer_image_id
            
            if answer_img_path.exists():
                with Image.open(answer_img_path) as ans_img:
                    contents[0]["parts"].append({"text": "- Solution Diagrams/Images:"})
                    contents[0]["parts"].append({"inline_data": ans_img})
            else:
                print(f"Answer image {answer_image_id} not found for {original_image_name}")
        # print(f"After Adding Images, Solution, Solution image and Rubric Token Count: {models["Gemini2.0Flash"].count_tokens(contents).total_tokens}")
        # Run each model
        for model_name, model in models.items():
            try:

                if model_name == 'Gemini2.5Pro':
                    response = type('obj', (object,), {'text': 'Cannot access', 'usage_metadata': None})()
                else:
                    response = model.generate_content(contents)
                
                # Store results
                outputs[f'{model_name}_response'] = response.text
                
                # Extract token counts
                if hasattr(response, 'usage_metadata') and hasattr(response.usage_metadata, 'prompt_token_count'):
                    input_tokens[f'{model_name}_input_tokens'] = response.usage_metadata.prompt_token_count
                    output_tokens[f'{model_name}_output_tokens'] = response.usage_metadata.candidates_token_count
                else:
                    input_tokens[f'{model_name}_input_tokens'] = 0
                    output_tokens[f'{model_name}_output_tokens'] = 0
                    
            except Exception as e:
                print(f"Error with {model_name} for {original_image_name}: {str(e)}")
                outputs[f'{model_name}_response'] = f"ERROR: {str(e)}"
                input_tokens[f'{model_name}_input_tokens'] = 0
                output_tokens[f'{model_name}_output_tokens'] = 0
        
        # Combine all data
        combined_row = row.to_dict()
        combined_row.update(outputs)
        combined_row.update(input_tokens)
        combined_row.update(output_tokens)
        
        # Add to subject results
        subject_results.append(combined_row)
        
        # Save incrementally to prevent data loss
        pd.DataFrame(subject_results).to_csv(f'{subject}_gemini_results_temp.csv', index=False)
    
    # Save final results for this subject
    if subject_results:
        output_file = f'{subject}_gemini_results.csv'
        pd.DataFrame(subject_results).to_csv(output_file, index=False)
        print(f"Saved {len(subject_results)} results to {output_file}")
    else:
        print(f"No results for {subject}")

print("Processing complete for all subjects!")