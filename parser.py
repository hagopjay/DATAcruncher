!pip install pandas matplotlib zipfile2 tqdm


import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# Specify your storage directory and output folder for processed metadata
data_directory = '/path/to/your/storage'
output_directory = '/path/to/save/extracted_data'
os.makedirs(output_directory, exist_ok=True)

# Function to extract metadata from each file
def extract_metadata(zip_file_path):
    metadata = {}
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            metadata[file_info.filename] = {
                'FileName': file_info.filename,
                'CompressedSize': file_info.compress_size,
                'UncompressedSize': file_info.file_size,
                'LastModified': datetime(*file_info.date_time),
            }
    return pd.DataFrame.from_dict(metadata, orient='index')

# Process all zip files in the data directory
all_metadata = pd.DataFrame()
for zip_file in tqdm(os.listdir(data_directory)):
    if zip_file.endswith('.zip'):
        zip_path = os.path.join(data_directory, zip_file)
        metadata_df = extract_metadata(zip_path)
        metadata_df['SourceZip'] = zip_file  # Track which zip each file came from
        all_metadata = pd.concat([all_metadata, metadata_df], ignore_index=True)

# Save metadata to CSV for easy access later
metadata_file = os.path.join(output_directory, 'all_metadata.csv')
all_metadata.to_csv(metadata_file, index=False)
print(f"Metadata saved to {metadata_file}")

# Visualizations and Basic Analysis
def plot_metadata_statistics(metadata_df):
    # Plot distribution of compressed vs uncompressed file sizes
    plt.figure(figsize=(10, 5))
    plt.hist(metadata_df['CompressedSize'] / 1024 / 1024, bins=50, alpha=0.7, label='Compressed Size (MB)')
    plt.hist(metadata_df['UncompressedSize'] / 1024 / 1024, bins=50, alpha=0.7, label='Uncompressed Size (MB)')
    plt.xlabel('Size (MB)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Distribution of Compressed and Uncompressed File Sizes')
    plt.show()

    # Plot file counts per zip
    zip_file_counts = metadata_df['SourceZip'].value_counts()
    plt.figure(figsize=(10, 5))
    zip_file_counts.plot(kind='bar')
    plt.xlabel('Zip File')
    plt.ylabel('Number of Files')
    plt.title('Number of Files per Zip Archive')
    plt.show()

# Run the visualization function
plot_metadata_statistics(all_metadata)
