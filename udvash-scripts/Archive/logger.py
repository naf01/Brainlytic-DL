import google.generativeai as genai
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import io
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"gemini_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GeminiProcessor")

# Configure Gemini API - Get API key from environment variable for security
API_KEY = "AIzaSyCtP2qSRxEucqbxAvuZ1yEbSBLVYX_Cazw"
if not API_KEY:
    logger.error("No API key found. Please set GEMINI_API_KEY environment variable.")
    raise ValueError("No API key found. Please set GEMINI_API_KEY environment variable.")

genai.configure(api_key=API_KEY)

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

# Initialize the models - Consider using a config file or environment variables
# to select which models to use
MODELS_TO_USE = ["Gemini2.5Flash-thinking"]  # Change this to select which models to run

models = {
    'Gemini2.0Flash': genai.GenerativeModel('gemini-2.0-flash'),
    'Gemini2.5Flash-thinking': genai.GenerativeModel('gemini-2.5-flash-preview-04-17-thinking'),
    'Gemini2.5Pro': genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
}

# Create a filtered model dictionary
selected_models = {k: v for k, v in models.items() if k in MODELS_TO_USE} if MODELS_TO_USE else models

class GeminiProcessor:
    def __init__(self, csv_path, base_dir, subjects=None):
        """Initialize the processor with paths and configuration"""
        print(csv_path, base_dir, subjects)
        self.df = pd.read_csv(csv_path).sort_values(by='image_filename', ascending=True).dropna(subset=['tags1'])
        self.base_dir = Path(base_dir)
        self.subjects = subjects or ['Physics', 'Chemistry', 'Math']
        logger.info(f"Initialized processor with {len(self.df)} entries and subjects: {self.subjects}")
        
    def process_all_subjects(self):
        """Process all configured subjects"""
        for subject in self.subjects:
            self.process_subject(subject)
        logger.info("Processing complete for all subjects!")
            
    def process_subject(self, subject):
        """Process a single subject"""
        logger.info(f"Processing {subject}...")
        
        # Path to unevaluated images for this subject
        unevaluated_dir = self.base_dir / subject / 'Question Images' / 'Unevaluated Images'
        
        # Path to answer images for this subject
        answer_dir = self.base_dir / subject / 'Answer Images'
        
        # Skip if directory doesn't exist
        if not unevaluated_dir.exists():
            logger.warning(f"Directory {unevaluated_dir} does not exist, skipping...")
            return
            
        # Get all unevaluated images
        unevaluated_images = list(unevaluated_dir.glob('*.jpg'))
        unevaluated_images.sort()
        logger.info(f"Found {len(unevaluated_images)} images to process for {subject}")
        
        # Create a list to store results for this subject
        subject_results = []
        
        # Process each unevaluated image
        for img_path in tqdm(unevaluated_images, desc=f"Processing {subject} images"):
            # Process single image and get results
            result = self.process_image(img_path, answer_dir)
            if result:
                subject_results.append(result)
                # Save incrementally to prevent data loss
                self._save_temp_results(subject_results, subject)
        
        # Save final results for this subject
        if subject_results:
            output_file = f'{subject}_gemini_results.csv'
            pd.DataFrame(subject_results).to_csv(output_file, index=False)
            logger.info(f"Saved {len(subject_results)} results to {output_file}")
        else:
            logger.warning(f"No results for {subject}")
            
    def process_image(self, img_path, answer_dir):
        """Process a single image and return results"""
        # Get original image filename (without '_reconstructed' if present)
        if '_reconstructed' in img_path.stem:
            original_image_name = img_path.name.replace('_reconstructed', '')
        else:
            original_image_name = img_path.name
        # Find matching row in the dataframe
        row_match = self.df[(self.df['image_filename'] == original_image_name)]
        #logging.info(f"Searching for image: {original_image_name} in CSV and {self.df['image_filename']}")
        if row_match.empty:
            logger.warning(f"Image {original_image_name} not found in CSV, skipping...")
            return None
        logging.info(f"Processing image: {original_image_name}")
        row = row_match.iloc[0]
        # Get solution text and rubric
        solve_text = row.get('solve_text', '')
        rubric = row.get('rubric', '')
        solution_image_id = row.get('solution_image_id', None)
        
        # Create prompt contents
        contents = self._create_prompt_contents(
            img_path, 
            solve_text, 
            rubric, 
            solution_image_id, 
            answer_dir
        )
        
        if not contents:
            logger.warning(f"Failed to create contents for {original_image_name}")
            return None
        
        # Run all models and collect results
        outputs, input_tokens, output_tokens = self._run_models(contents, original_image_name)
        
        # Combine all data
        combined_row = row.to_dict()
        combined_row.update(outputs)
        combined_row.update(input_tokens)
        combined_row.update(output_tokens)
        
        return combined_row
    
    def _create_prompt_contents(self, img_path, solve_text, rubric, solution_image_id, answer_dir):
        """Create a properly formatted prompt with images"""
        try:
            # Initialize content structure
            contents = [{"role": "user", "parts": [{"text": SYSTEM_INSTRUCTIONS}]}]
            
            # Log base token count
            base_token_count = selected_models[list(selected_models.keys())[0]].count_tokens(
                SYSTEM_INSTRUCTIONS
            ).total_tokens
            logger.info(f"Base prompt token count: {base_token_count}")
        
            # Add student image with clear labeling
            with Image.open(img_path) as script_img:
                # Convert image to bytes
                img_byte_arr = io.BytesIO()
                script_img.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()
                
                contents[0]["parts"].append({"text": "- Student's Script Image (analyze this for evaluation):"})
                contents[0]["parts"].append({
                    "inline_data": {
                        "mime_type": "image/jpeg", 
                        "data": img_bytes
                    }
                })
            
            # Log token count after adding student image
            after_student_image_count = selected_models[list(selected_models.keys())[0]].count_tokens(
                contents
            ).total_tokens
            logger.info(f"After adding student image token count: {after_student_image_count} " +
                      f"(+{after_student_image_count - base_token_count})")
            
            # Add solution text
            contents[0]["parts"].append({"text": "- Solution (compare with student's work):"})
            contents[0]["parts"].append({"text": str(solve_text) if pd.notna(solve_text) else "No solution text provided"})
            
            # Add rubric
            contents[0]["parts"].append({"text": "- Marking Rubric (use to guide evaluation):"})
            contents[0]["parts"].append({"text": str(rubric) if pd.notna(rubric) else "No rubric provided"})
            
            # Log token count after adding text content
            after_text_content_count = selected_models[list(selected_models.keys())[0]].count_tokens(
                contents
            ).total_tokens
            logger.info(f"After adding solution and rubric token count: {after_text_content_count} " +
                      f"(+{after_text_content_count - after_student_image_count})")
            
            # Add answer image if available
            if pd.notna(solution_image_id):
                answer_image_id = f"{int(solution_image_id)}.png"
                answer_img_path = answer_dir / answer_image_id
                
                if answer_img_path.exists():
                    with Image.open(answer_img_path) as ans_img:
                        # Convert answer image to bytes
                        ans_byte_arr = io.BytesIO()
                        ans_img.save(ans_byte_arr, format='PNG')
                        ans_bytes = ans_byte_arr.getvalue()
                        
                        contents[0]["parts"].append({"text": "- Solution Diagram/Image (reference):"})
                        contents[0]["parts"].append({
                            "inline_data": {
                                "mime_type": "image/png", 
                                "data": ans_bytes
                            }
                        })
                        
                    # Log token count after adding answer image
                    final_token_count = selected_models[list(selected_models.keys())[0]].count_tokens(
                        contents
                    ).total_tokens
                    logger.info(f"After adding solution image token count: {final_token_count} " +
                              f"(+{final_token_count - after_text_content_count})")
                else:
                    logger.warning(f"Answer image {answer_image_id} not found")
            
            return contents
            
        except Exception as e:
            logger.error(f"Error creating prompt: {str(e)}")
            return None
    
    def _run_models(self, contents, image_name):
        """Run all selected models on contents and return results"""
        outputs = {}
        input_tokens = {}
        output_tokens = {}
        
        for model_name, model in selected_models.items():
            try:
                logger.info(f"Running model {model_name} for {image_name}")
    
                
                # Generate response
                response = model.generate_content(
                    contents, 
                )
                
                # Store results
                outputs[f'{model_name}_response'] = response.text
                
                # Extract token counts
                if hasattr(response, 'usage_metadata') and hasattr(response.usage_metadata, 'prompt_token_count'):
                    input_tokens[f'{model_name}_input_tokens'] = response.usage_metadata.prompt_token_count
                    output_tokens[f'{model_name}_output_tokens'] = response.usage_metadata.candidates_token_count
                    logger.info(f"Model {model_name} used {response.usage_metadata.prompt_token_count} input tokens " +
                              f"and {response.usage_metadata.candidates_token_count} output tokens")
                else:
                    input_tokens[f'{model_name}_input_tokens'] = 0
                    output_tokens[f'{model_name}_output_tokens'] = 0
                    
            except Exception as e:
                logger.error(f"Error with {model_name} for {image_name}: {str(e)}")
                outputs[f'{model_name}_response'] = f"ERROR: {str(e)}"
                input_tokens[f'{model_name}_input_tokens'] = 0
                output_tokens[f'{model_name}_output_tokens'] = 0
        
        return outputs, input_tokens, output_tokens
    
    def _save_temp_results(self, results, subject):
        """Save intermediate results to prevent data loss"""
        temp_file = f'{subject}_gemini_results_temp_with_thinking.csv'
        pd.DataFrame(results).to_csv(temp_file, index=False)
        logger.debug(f"Saved {len(results)} temporary results to {temp_file}")

# Main execution
if __name__ == "__main__":
    # Configure paths and settings
    CSV_PATH = 'Gemini_Math.csv'
    BASE_DIR = 'UDV'
    SUBJECTS = ['Math']  # Add subjects as needed
    # Create processor and run
    processor = GeminiProcessor(CSV_PATH, BASE_DIR, SUBJECTS)
    processor.process_all_subjects()