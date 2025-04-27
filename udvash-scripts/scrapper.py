import os
import csv
import requests
from bs4 import BeautifulSoup
import base64
import shutil
import json
from urllib.parse import urljoin, urlparse

def extract_text_from_image(image_path):
    api_key = "AIzaSyBSISlUTjixUxp6_eIAHnMEIdVmTHnlyhE"
    endpoint = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"

    # Read and encode the image
    with open(image_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode("utf-8")

    # Prepare the request payload
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": "image/png",  # <-- since your file is .png
                            "data": encoded_image
                        }
                    },
                    {
                        "text": """extract the text excluding the image. give everything in well formatted structure. e.g. for equations use (/a/, /fraction{}{} etc. give the text in bash shell. don't give explanation or comment."""
                    }
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json"
    }

    # Send request
    response = requests.post(endpoint, headers=headers, data=json.dumps(payload))

    # Parse response
    if response.status_code == 200:
        data = response.json()
        try:
            return data['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError):
            return "Error: Unexpected response format."
    else:
        return f"Error: {response.status_code} - {response.text}"
    


def download_webpage_with_assets(save_folder="downloaded_page", url=None):
    if not url:
        raise ValueError("URL is required to download the webpage.")

    headers = {
        "Host": "online.udvash-unmesh.com",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.50 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    }
    
    # Make the request
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch page, status code {response.status_code}")
    
    # Create save folder
    os.makedirs(save_folder, exist_ok=True)
    
    # Parse HTML
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Download assets (images, SVGs, etc.)
    asset_tags = soup.find_all(["img", "link", "script"])
    for tag in asset_tags:
        asset_url = None
        if tag.name == "img" and tag.get("src"):
            asset_url = tag["src"]
        elif tag.name == "link" and tag.get("href"):
            asset_url = tag["href"]
        elif tag.name == "script" and tag.get("src"):
            asset_url = tag["src"]
        
        if asset_url:
            asset_full_url = urljoin(url, asset_url)
            try:
                asset_response = requests.get(asset_full_url, headers=headers)
                if asset_response.status_code == 200:
                    parsed_url = urlparse(asset_url)
                    asset_path = os.path.join(save_folder, os.path.basename(parsed_url.path))
                    with open(asset_path, "wb") as f:
                        f.write(asset_response.content)
                    
                    # Update tag to point to local file
                    if tag.name == "link":
                        tag["href"] = os.path.basename(parsed_url.path)
                    else:
                        tag["src"] = os.path.basename(parsed_url.path)
            except Exception as e:
                print(f"Failed to download")
    
    # Save modified HTML
    html_path = os.path.join(save_folder, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(str(soup))
    
    print(f"Page downloaded successfully into")



def parse_and_export(html_path, output_csv='results.csv', images_dir='images', start_id=1005):
    """
    Parse the downloaded HTML file at html_path, extract marks, image URLs, solutions,
    download images to images_dir with IDs starting from start_id, and save data to output_csv.
    """
    # Check if the HTML file exists
    if not os.path.exists(html_path):
        raise FileNotFoundError(f"The HTML file '{html_path}' does not exist.")
    
    # Create the images directory if it doesn't exist
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # Read the HTML file
    with open(html_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    
    # Find all the relevant sections in the HTML
    # Find all <div class="row questionBlock"> elements
    question_blocks = soup.find_all('div', class_='row questionBlock')
    
    # Filter question blocks containing the specific span with "Request for Review" or green score span
    filtered_blocks = [
        block for block in question_blocks
        if (
            block.find('span', class_='reviewRequest', id='', style='margin-bottom:10px;') and
            'Request for Review' in block.find('span', class_='reviewRequest', id='', style='margin-bottom:10px;').text
        ) or (
            block.find('span', style='color:green; font-weight:bold; float:right !important;') and
            block.find('span', style='color:green; font-weight:bold; float:right !important;').text.strip().endswith('/10')
        )
    ]
    
    # Print the number of filtered blocks
    print(f"Filtered {len(filtered_blocks)} question blocks with 'Correct'.")

    # Check if the whole HTML contains any of the specified strings
    html_text = soup.get_text()
    subjects_found = [subject for subject in ["Math", "Physics", "Chemistry"] if subject in html_text]

    # Determine the subject name based on the found strings and total blocks
    total_blocks = len(filtered_blocks)
    if len(subjects_found) == 3:
        if total_blocks <= 18:
            subject_mapping = {
                range(1, 7): "Math",
                range(7, 13): "Physics",
                range(13, 19): "Chemistry"
            }
        elif total_blocks <= 40 and total_blocks > 18:
            subject_mapping = {
                range(1, 15): "Math",
                range(15, 28): "Physics",
                range(28, 41): "Chemistry"
            }
        else:
            subject_mapping = {
                range(1, 100): "Science"  # Default to Math for all questions if all subjects are found
            }
    elif len(subjects_found) == 1:
        subject_mapping = {range(1, 1000): subjects_found[0]}  # Apply the single subject to all questions
    else:
        if total_blocks > 18 and total_blocks <= 40:
            subject_mapping = {
                range(1, 15): "Math",
                range(15, 28): "Physics",
                range(28, 41): "Chemistry"
            }
        else:
            subject_mapping = {
                range(1, 7): "Math",
                range(7, 13): "Physics",
                range(13, 19): "Chemistry"
            }

    print(f"Subjects found: {subjects_found}")

    # Initialize a list to store data for CSV
    data = []

    # Process each filtered block
    for idx, block in enumerate(filtered_blocks, start=1):
        # Find the image tag
        img_tag = block.find('img', crossorigin='anonymous')
        if img_tag and 'src' in img_tag.attrs:
            # Extract the image source
            img_src = img_tag['src']
            
            # Construct the source path (relative to the HTML file location)
            img_path = os.path.join(os.path.dirname(html_path), img_src)
            # Check for a solution image in base64 format
            solution_img_tag = block.find('img', src=lambda x: x and x.startswith('data:image/'))
            solution_image_id = None
            if solution_img_tag:
                # Extract the base64 string
                base64_data = solution_img_tag['src'].split(',')[1]
                
                # Decode the base64 string
                solution_image_data = base64.b64decode(base64_data)
                
                # Construct the filename for the solution image
                solution_image_filename = f"{start_id}.png"
                solution_image_path = os.path.join(images_dir, solution_image_filename)
                
                # Save the decoded image to the file
                with open(solution_image_path, 'wb') as solution_file:
                    solution_file.write(solution_image_data)
                
                # Store the solution image ID
                solution_image_id = start_id
                
                # Increment the ID for the next image
                start_id += 1
            # Check if the image file exists
            if os.path.exists(img_path):
                # Construct the destination path
                img_filename = f"{start_id}.jpg"
                img_dest_path = os.path.join(images_dir, img_filename)
                
                # Copy the image to the destination directory
                with open(img_path, 'rb') as src_file:
                    with open(img_dest_path, 'wb') as dest_file:
                        dest_file.write(src_file.read())
                
                start_id += 1
                
                # Extract the question ID from the block
                question_id_tag = block.find(string=lambda text: text and text.startswith("Question "))
                if question_id_tag:
                    question_id = question_id_tag.split("Question ")[-1].strip()
                else:
                    question_id = None
                
                # Extract the mark from the block by searching for '/10' in the block's text
                block_text = block.get_text()
                if '/10' in block_text:
                    mark = block_text.split('/10')[0].strip().split()[-1]
                else:
                    mark = None
                # Extract the solution text from the solveText div
                solve_text_div = block.find('div', class_='row solveText', style=' display: block; padding:10px; color:#077307; overflow-x: auto; background-color: #d7f7d4; ')
                if solve_text_div:
                    # Extract the HTML content including the <div> tag itself
                    solve_text = str(solve_text_div)
                else:
                    solve_text = None
                    
                # Determine the subject name based on the question ID
                subject_name = None
                if question_id:
                    question_id = int(question_id)
                    for question_range, subject in subject_mapping.items():
                        if question_id in question_range:
                            subject_name = subject
                            break

                # Store the data for this block
                # ScriptID,Script Url,Marks,Subject,Solution
                data.append({
                    'question_id': question_id, # Question ID
                    'image_filename': img_filename, # ScriptID
                    'Marks': mark, # Marks
                    'solution_image_id': solution_image_id, # Solution Image ID
                    'solve_text': solve_text, # Solution
                    'subject': subject_name, # Subject
                })
                
    for i in range(len(data)):
        # Load the temporary HTML file
        with open('template.html', 'r', encoding='utf-8') as temp_file:
            temp_soup = BeautifulSoup(temp_file, 'html.parser')

        # Find the <div class="TakeExamBody"> inside <div class="TakeExamWrapper">
        take_exam_body_div = temp_soup.find('div', class_='TakeExamWrapper').find('div', class_='TakeExamBody')

        if take_exam_body_div and data:
            # Replace the content of the TakeExamBody div with the solve_text from data[0]
            take_exam_body_div.clear()
            take_exam_body_div.append(BeautifulSoup(data[i]['solve_text'], 'html.parser'))

            # Save the modified HTML back to try.html
            with open('template.html', 'w', encoding='utf-8') as temp_file:
                temp_file.write(str(temp_soup))

            # Render the modified HTML and capture a snapshot using Selenium WebDriver

            # Set up Selenium WebDriver with headless Chrome
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920x1080")
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

            import time
            from requests.exceptions import ConnectionError

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Load the modified HTML file in the browser
                    driver.get(f"file://{os.path.abspath('template.html')}")

                    # Define the output path for the rendered image
                    rendered_image_path = os.path.join('rendered_images', f"i_{i}.png")
                    if not os.path.exists('rendered_images'):
                        os.makedirs('rendered_images')

                    # Capture a screenshot of the page
                    driver.save_screenshot(rendered_image_path)
                    break  # Exit the retry loop if successful
                except ConnectionError as e:
                    print(f"ConnectionError occurred. Retrying in 10 seconds...")
                    time.sleep(10)
                except Exception as e:
                    print(f"An unexpected error occurred. Retrying in 10 seconds...")
                    time.sleep(10)
                finally:
                    # Close the WebDriver
                    driver.quit()

        # Extract text from the rendered image and update the data dictionary
        extracted_text = extract_text_from_image(rendered_image_path)
        # Remove the bash formatting and keep only the text
        extracted_text = extracted_text.replace("```bash", "").replace("```", "").strip()
        data[i]['solve_text'] = extracted_text

        # Delete the rendered image after processing
        # if os.path.exists(rendered_image_path):
        #     os.remove(rendered_image_path)

    # Convert the data list to a pandas DataFrame
    df = pd.DataFrame(data)

    # Check if the CSV file already exists
    if os.path.exists('evaluation_results.csv'):
        # Load the existing data
        existing_df = pd.read_csv('evaluation_results.csv', encoding='utf-8')
        # Append the new data to the existing data
        df = pd.concat([existing_df, df], ignore_index=True)
    
    # Save the updated DataFrame to the CSV file
    df.to_csv('evaluation_results.csv', index=False, encoding='utf-8')

    # Print the collected data for debugging
    print(f"Collected data for {len(data)} question blocks.")

if __name__ == '__main__':
    import argparse
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
    import pandas as pd

    parser = argparse.ArgumentParser(description='Extract marks, images, and solutions from a provided HTML file.')
    parser.add_argument('html_path', help='Path to the HTML file to process')
    parser.add_argument('--csv', default='results.csv', help='Output CSV filename')
    parser.add_argument('--images', default='images', help='Directory to save extracted images')
    
    try:
        args = parser.parse_args()
    except SystemExit:
        parser.print_help()
        raise

    start_id = 1001

    # Determine the starting ID by checking the latest file in the images directory
    if os.path.exists(args.images):
        image_files = [f for f in os.listdir(args.images) if f.endswith(('.jpg', '.png'))]
        if image_files:
            latest_file = max(image_files, key=lambda x: int(os.path.splitext(x)[0]))
            start_id = int(os.path.splitext(latest_file)[0]) + 1
        else:
            start_id = 1001  # Default start ID if no images are present
    else:
        start_id = 1001  # Default start ID if the directory doesn't exist

    try:
        # Process the provided HTML file
        parse_and_export(args.html_path, args.csv, args.images, start_id)
    finally:
        # Delete the provided HTML file after processing
        if os.path.exists(args.html_path):
            # os.remove(args.html_path)
            # # Delete the provided HTML file after processing
            # if os.path.exists(args.html_path):
            #     os.remove(args.html_path)
            
            # # # Remove the directory associated with the HTML file (e.g., "1_files" for "1.html")
            # html_dir = f"{os.path.splitext(args.html_path)[0]}_files"
            # if os.path.exists(html_dir) and os.path.isdir(html_dir):
            #     shutil.rmtree(html_dir)
            pass