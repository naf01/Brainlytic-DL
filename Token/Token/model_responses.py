import google.generativeai as genai
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import random
genai.configure(api_key="AIzaSyBSISlUTjixUxp6_eIAHnMEIdVmTHnlyhE")

SYSTEM_INSTRUCTIONS_1 = """

ou are a script checker who excels in evaluating and reviewing exam scripts in Bengali. Your response should be in Bengali too. You will be provided only an image of a student’s handwritten script. At the top of the image is the problem statement, followed by the student’s written solution. 

Develop your understanding of the problem statement and the image context first and relate to the solution properly. 
Your job is to evaluate the script fairly, awarding a score out of 10. 
Make sure you correctly grasp hand writing no matter how worse it is. Extrapolate from the surrounding if any letter or word is incomplete.

Pinpoint any mistakes clearly and explain why they are incorrect. Do not repeat the full corrected steps; only highlight what was wrong. Provide concise, constructive remarks for the student at the end.

You will be provided:
Script Image

Response Structure:

মার্কস: (total marks, and then showing how much mark is given for which correct step)
চিহ্নিত ভুল: (bulleted and concise)
ভুল গুলোর সঠিক: (bulleted and concise)
অসম্পূর্ণ উত্তরের ক্ষেত্রে বাকি অংশের সমাধান: (concise)
মার্কস কাটার কারণ: (bulleted and concise, showing how much mark is deducted for which mistake)

"""

SYSTEM_INSTRUCTIONS_2 = """
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

models = {
    'Gemini2.0Flash': genai.GenerativeModel('gemini-2.0-flash'),
    'Gemini2.5Flash': genai.GenerativeModel('gemini-2.5-flash-preview-04-17'),
    'Gemini2.5Pro': genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
}


df = pd.read_csv('Udvash Data - Final.csv')



folders = ['Math', 'Chemistry', 'Physics']
all_files = []
for folder in folders[:1]:
    all_files.extend(Path(folder).glob('*_reconstructed.jpg'))

answer_path = Path('Answer Images')
answer_images = answer_path.glob('*.png')

# Randomly select 15 items from all_files
all_files = random.sample(all_files, min(len(all_files), 15))


results = []

for img_path in tqdm(all_files, desc="Processing images"):
    image_id = img_path.name.replace('_reconstructed.jpg', '.jpg')
    
    row_match = df[df['image_filename'] == image_id]
    if row_match.empty:
        print(f"Image {image_id} not found in CSV, skipping...")
        continue
    row = row_match.iloc[0]
    
    solve_text = row['solve_text']
    answer_image = row['solution_image_id']

    outputs = {}
    tokens = {}
    
    prompt1 = SYSTEM_INSTRUCTIONS_1
    prompt2 = SYSTEM_INSTRUCTIONS_2.replace('{Solution}', solve_text)

    img_files = []
    with Image.open(img_path) as script_img:
        img_files.append(script_img.copy())  

    if pd.notna(answer_image):
        answer_image_id = str(int(answer_image)) + '.png'  
        answer_img_path = answer_path / answer_image_id
        if answer_img_path.exists():
            with Image.open(answer_img_path) as ans_img:
                img_files.append(ans_img.copy())
        else:
            print(f"Answer image {answer_image} not found, skipping answer image.")

    for model_name, model in models.items():
        response1 = model.generate_content([prompt1] + img_files)
        outputs[model_name + '_1'] = response1.text
        tokens[model_name + '_1_Token'] = response1.usage_metadata.candidates_token_count
        #print(f'[{model_name} - Prompt 1] Response:')
        #print(response1.text)

        response2 = model.generate_content([prompt2] + img_files)
        outputs[model_name + '_2'] = response2.text
        tokens[model_name + '_2_Token'] = response2.usage_metadata.candidates_token_count
        #print(f'[{model_name} - Prompt 2] Response:')
        #print(response2.text)
        #print('---')

    combined_row = row.to_dict()
    combined_row.update(outputs)
    combined_row.update(tokens)
    
    results.append(combined_row)
    pd.DataFrame(results).to_csv('gemini_results.csv', index=False)