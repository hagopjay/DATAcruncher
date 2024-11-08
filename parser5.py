import os
import zipfile
import json
from tqdm import tqdm

# Specify your storage directory and output folder for extracted non-media JSON files
data_directory = '/path/to/your/storage'
non_media_json_output_directory = '/path/to/save/extracted_non_media_json_files'
os.makedirs(non_media_json_output_directory, exist_ok=True)

# Define media extensions to check for within JSON files
media_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mov', '.m4v', '.mp3', '.heic', 'hvac' ]

# Function to check if a JSON file contains media references (case insensitive)
def contains_media_references(json_content):
    json_content_lower = json_content.lower()  # Convert content to lowercase for case-insensitive checking
    for ext in media_extensions:
        if ext in json_content_lower:
            return True
    return False

# Function to extract only JSON and JSONL files that do not contain media references
def extract_non_media_json_files(zip_file_path, output_dir):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            file_name = file_info.filename
            file_extension = os.path.splitext(file_name)[-1].lower()
            
            # Check if the file is JSON or JSONL
            if file_extension in ['.json', '.jsonl']:
                with zip_ref.open(file_name) as file:
                    try:
                        # Load JSON content
                        json_content = file.read().decode('utf-8')
                        
                        # Skip files containing media references
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

# Process each zip file and extract non-media JSON/JSONL files
for zip_file in tqdm(os.listdir(data_directory)):
    if zip_file.endswith('.zip'):
        zip_path = os.path.join(data_directory, zip_file)
        extract_non_media_json_files(zip_path, non_media_json_output_directory)

print(f"All non-media JSON and JSONL files have been extracted to {non_media_json_output_directory}")
