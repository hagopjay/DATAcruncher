import os
import zipfile
import json
from tqdm import tqdm

# Specify your storage directory and output folder for extracted JSON files
data_directory = './'
json_output_directory = './extracted_json_files'
os.makedirs(json_output_directory, exist_ok=True)

# Define media extensions to check for within JSON files
media_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mov', '.m4v', '.mp3', '.heic']

# Function to check if a JSON file references media files
def contains_media_references(json_content):
    for ext in media_extensions:
        if ext in json_content:
            return True
    return False

# Function to extract JSON and JSONL files that don't contain media references
def extract_filtered_json_files(zip_file_path, output_dir):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            file_name = file_info.filename
            file_extension = os.path.splitext(file_name)[-1].lower()
            
            # Check if file is JSON or JSONL
            if file_extension in ['.json', '.jsonl']:
                # Read JSON content without extracting if it contains media references
                with zip_ref.open(file_name) as file:
                    try:
                        # Load JSON content
                        json_content = file.read().decode('utf-8')
                        if contains_media_references(json_content):
                            print(f"Skipped: {file_name} (contains media references)")
                            continue
                        
                        # Save JSON files without media references to the output directory
                        target_path = os.path.join(output_dir, os.path.basename(file_name))
                        with open(target_path, 'w') as target:
                            target.write(json_content)
                        print(f"Extracted: {file_name} from {os.path.basename(zip_file_path)}")
                    
                    except json.JSONDecodeError:
                        print(f"Skipped: {file_name} (invalid JSON format)")

# Process each zip file and extract JSON/JSONL files without media references
for zip_file in tqdm(os.listdir(data_directory)):
    if zip_file.endswith('.zip'):
        zip_path = os.path.join(data_directory, zip_file)
        extract_filtered_json_files(zip_path, json_output_directory)

print(f"All JSON and JSONL files without media references have been extracted to {json_output_directory}")
