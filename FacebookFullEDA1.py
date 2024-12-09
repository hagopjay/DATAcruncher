import os
import json
import zipfile
from collections import defaultdict, Counter
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Any, Union
import logging
import re
from pathlib import Path


class ZipMapper:
    def __init__(self):
        self.structure = {}
        self.indent = "  "
    
    def _format_size(self, size_in_bytes):
        """Convert bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_in_bytes < 1024:
                return f"{size_in_bytes:.1f}{unit}"
            size_in_bytes /= 1024
        return f"{size_in_bytes:.1f}TB"

    def map_zip(self, zip_path, current_depth=0, max_depth=10):
        """Map the contents of a ZIP file recursively"""
        if current_depth >= max_depth:
            return {"error": "Max depth exceeded"}
        
        result = {
            "name": os.path.basename(zip_path),
            "type": "zip",
            "size": os.path.getsize(zip_path),
            "contents": {}
        }
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for info in zf.infolist():
                    path_parts = info.filename.split('/')
                    current_dict = result["contents"]
                    
                    # Handle directory structure
                    for part in path_parts[:-1]:
                        if part not in current_dict:
                            current_dict[part] = {"type": "directory", "contents": {}}
                        current_dict = current_dict[part]["contents"]
                    
                    # Handle file
                    filename = path_parts[-1]
                    if filename:  # Skip empty filenames (pure directories)
                        file_data = {
                            "type": "file",
                            "size": info.file_size,
                            "compressed_size": info.compress_size,
                            "date": f"{info.date_time[0]}-{info.date_time[1]:02d}-{info.date_time[2]:02d}"
                        }
                        
                        # If it's a nested ZIP, try to map it too
                        if filename.endswith('.zip'):
                            try:
                                with zf.open(info.filename) as nested_zip_data:
                                    temp_path = f"temp_nested_{filename}"
                                    with open(temp_path, 'wb') as temp_zip:
                                        temp_zip.write(nested_zip_data.read())
                                    
                                    file_data["contents"] = self.map_zip(temp_path, 
                                                                       current_depth + 1, 
                                                                       max_depth)["contents"]
                                    os.remove(temp_path)
                            except Exception as e:
                                file_data["error"] = str(e)
                        
                        current_dict[filename] = file_data
                        
        except Exception as e:
            result["error"] = str(e)
            
        return result



    def generate_ascii_tree(self, structure, prefix="", is_last=True):
        """Generate ASCII tree representation of the ZIP structure"""
        output = []
        
        # Add current node
        connector = "└── " if is_last else "├── "
        name = structure.get("name", "")
        size = self._format_size(structure.get("size", 0))
        node_info = f"{name} ({size})" if name else ""
        
        if node_info:
            output.append(prefix + connector + node_info)
        
        # Handle contents
        contents = structure.get("contents", {})
        items = list(contents.items())
        
        for i, (name, item) in enumerate(items):
            is_last_item = i == len(items) - 1
            new_prefix = prefix + ("    " if is_last else "│   ")
            
            # Add size information to files
            if item.get("type") == "file":
                size = self._format_size(item.get("size", 0))
                date = item.get("date", "")
                item_text = f"{name} ({size}) [{date}]"
            else:
                item_text = name
                
            output.append(new_prefix + ("└── " if is_last_item else "├── ") + item_text)
            
            # Recursively handle subdirectories or nested ZIPs
            if "contents" in item:
                output.extend(self.generate_ascii_tree(
                    {"contents": item["contents"]}, 
                    new_prefix, 
                    is_last_item
                ))
                
        return output



class FacebookDataAnalyzer:
    def __init__(self, data_directory: str, output_directory: str):
        self.data_directory = data_directory
        self.output_directory = output_directory
        self.metadata_df = pd.DataFrame()
        
        # Initialize data structures
        self.word_counts = Counter()
        self.text_data = []
        self.json_structure = defaultdict(set)
        self.timestamps = []
        self.email_patterns = defaultdict(int)
        self.location_data = []
        self.sentiment_scores = []
        self.file_metadata = []
        
        # Set up logging
        os.makedirs(output_directory, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_directory, 'facebook_data_analysis.log')),
                logging.StreamHandler()
            ]
        )
        
        # Download required NLTK data
        self._setup_nltk()

    def map_zip_contents(self):
        """Create a map of all ZIP file contents"""
        mapper = ZipMapper()
        zip_maps = {}
        
        zip_files = [f for f in os.listdir(self.data_directory) if f.endswith('.zip')]
        for zip_file in zip_files:
            zip_path = os.path.join(self.data_directory, zip_file)
            zip_maps[zip_file] = mapper.map_zip(zip_path)
            
            # Generate and save ASCII tree
            tree_lines = mapper.generate_ascii_tree(zip_maps[zip_file])
            tree_output = os.path.join(self.output_directory, f'{zip_file}_structure.txt')
            with open(tree_output, 'w', encoding='utf-8') as f:
                f.write('\n'.join(tree_lines))
                
        return zip_maps

    def _setup_nltk(self):
        """Download required NLTK data."""
        for package in ['stopwords', 'punkt', 'vader_lexicon']:
            try:
                nltk.download(package, quiet=True)
            except Exception as e:
                logging.error(f"Failed to download NLTK package {package}: {str(e)}")

    def process_file_content(self, file_content: str, file_path: str):
        """Process the content of a single file."""
        try:
            # Try to parse as JSON first
            if file_path.endswith(('.json', '.jsonl')):
                self.analyze_json_content(file_content, file_path)
            
            # Process as text regardless of format
            self.analyze_text_content(file_content)
            self.extract_timestamps(file_content)
            self.analyze_email_patterns(file_content)
            self.extract_location_data(file_content)
            self.analyze_sentiment(file_content)
            
        except Exception as e:
            logging.error(f"Error processing content from {file_path}: {str(e)}")

    def process_zip_file(self, zip_path: Union[str, Path], is_nested: bool = False):
        """Process a single zip file and its contents."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for file_info in zf.infolist():
                    if file_info.filename.endswith(('.json', '.jsonl', '.txt')):
                        try:
                            with zf.open(file_info.filename) as file:
                                content = file.read().decode('utf-8', errors='ignore')
                                self.process_file_content(content, file_info.filename)
                        except Exception as e:
                            logging.error(f"Error reading {file_info.filename}: {str(e)}")
                    
                    elif file_info.filename.endswith('.zip'):
                        # Handle nested zip
                        with zf.open(file_info.filename) as nested_zip_data:
                            temp_path = os.path.join(self.output_directory, 'temp_nested.zip')
                            with open(temp_path, 'wb') as temp_zip:
                                temp_zip.write(nested_zip_data.read())
                            
                            self.process_zip_file(temp_path, is_nested=True)
                            os.remove(temp_path)
                            
        except Exception as e:
            logging.error(f"Error processing zip file {zip_path}: {str(e)}")

    def analyze_all_data(self):
        """Main method to analyze all Facebook data."""
        # First, create a map of all ZIP contents
        zip_maps = self.map_zip_contents()
        
        # Process all zip files in the directory
        zip_files = [f for f in os.listdir(self.data_directory) if f.endswith('.zip')]
        
        for zip_file in tqdm(zip_files, desc="Processing Facebook data archives"):
            zip_path = os.path.join(self.data_directory, zip_file)
            
            # Extract and store metadata
            metadata_entries = self.extract_nested_zip_metadata(zip_path)
            self.metadata_df = pd.concat([
                self.metadata_df,
                pd.DataFrame(metadata_entries)
            ], ignore_index=True)
            
            # Process the actual content
            self.process_zip_file(zip_path)
        
        # Generate outputs
        self.save_metadata()
        self.generate_visualizations()
        self.generate_report()

    def extract_nested_zip_metadata(self, zip_path: Union[str, Path], parent_zip: str = None) -> List[dict]:
        """Extract metadata from potentially nested zip files."""
        metadata_entries = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for file_info in zf.infolist():
                    metadata = {
                        'FileName': file_info.filename,
                        'CompressedSize': file_info.compress_size,
                        'UncompressedSize': file_info.file_size,
                        'LastModified': datetime(*file_info.date_time),
                        'ParentZip': parent_zip or os.path.basename(str(zip_path)),
                        'IsNested': bool(parent_zip)
                    }
                    
                    metadata_entries.append(metadata)
                    
                    # Handle nested zip files
                    if file_info.filename.endswith('.zip'):
                        with zf.open(file_info.filename) as nested_zip_data:
                            temp_path = os.path.join(self.output_directory, 'temp_nested.zip')
                            with open(temp_path, 'wb') as temp_zip:
                                temp_zip.write(nested_zip_data.read())
                            
                            nested_metadata = self.extract_nested_zip_metadata(
                                temp_path, 
                                parent_zip=file_info.filename
                            )
                            metadata_entries.extend(nested_metadata)
                            
                            os.remove(temp_path)
                            
        except Exception as e:
            logging.error(f"Error processing zip file {zip_path}: {str(e)}")
            
        return metadata_entries

    def analyze_json_content(self, content: str, filename: str):
        """Analyze JSON content for structure and patterns."""
        try:
            data = json.loads(content)
            self._analyze_json_structure(data)
        except json.JSONDecodeError:
            logging.warning(f"Invalid JSON in {filename}")

    def _analyze_json_structure(self, data: Any, prefix: str = ''):
        """Recursively analyze JSON structure."""
        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                self.json_structure[new_prefix].add(type(value).__name__)
                self._analyze_json_structure(value, new_prefix)
        elif isinstance(data, list) and data:
            self._analyze_json_structure(data[0], f"{prefix}[]")

    def analyze_text_content(self, content: str):
        """Analyze text content for word frequencies and patterns."""
        words = word_tokenize(content.lower())
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.isalnum() and word not in stop_words]
        self.word_counts.update(words)
        self.text_data.append(content)

    def extract_timestamps(self, content: str):
        """Extract and analyze timestamp patterns."""
        timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO format
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # Common datetime format
            r'\d{2}/\d{2}/\d{4}'  # US date format
        ]
        
        for pattern in timestamp_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                try:
                    timestamp = pd.to_datetime(match.group())
                    self.timestamps.append(timestamp)
                except ValueError:
                    continue

    def analyze_email_patterns(self, content: str):
        """Analyze email communication patterns."""
        email_pattern = r'[\w\.-]+@[\w\.-]+'
        emails = re.findall(email_pattern, content)
        for email in emails:
            self.email_patterns[email] += 1

    def extract_location_data(self, content: str):
        """Extract and analyze location data."""
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                if 'latitude' in data and 'longitude' in data:
                    self.location_data.append((data['latitude'], data['longitude']))
                elif 'location' in data and isinstance(data['location'], dict):
                    loc = data['location']
                    if 'lat' in loc and 'lng' in loc:
                        self.location_data.append((loc['lat'], loc['lng']))
        except (json.JSONDecodeError, KeyError):
            pass

    def analyze_sentiment(self, content: str):
        """Analyze sentiment of text content."""
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        sentences = content.split('.')
        for sentence in sentences:
            if len(sentence.strip()) > 10:  # Only analyze non-trivial sentences
                score = sia.polarity_scores(sentence)
                self.sentiment_scores.append(score['compound'])

    def save_metadata(self):
        """Save the collected metadata to CSV."""
        metadata_path = os.path.join(self.output_directory, 'facebook_data_metadata.csv')
        self.metadata_df.to_csv(metadata_path, index=False)
        logging.info(f"Metadata saved to {metadata_path}")


    def generate_visualizations(self):
        """Generate various visualizations from the analyzed data."""
        # Create output directory for visualizations
        viz_dir = os.path.join(self.output_directory, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        # 1. Word Cloud
        plt.figure(figsize=(15, 8))
        wordcloud = WordCloud(width=1600, height=800, background_color='white').generate_from_frequencies(self.word_counts)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(os.path.join(viz_dir, 'wordcloud.png'))
        plt.close()

        # 2. Temporal Analysis
        if self.timestamps:
            df_timestamps = pd.DataFrame(self.timestamps, columns=['timestamp'])
            plt.figure(figsize=(15, 8))
            df_timestamps.groupby(df_timestamps['timestamp'].dt.year).size().plot(kind='bar')
            plt.title('Activity Over Time')
            plt.xlabel('Year')
            plt.ylabel('Number of Events')
            plt.savefig(os.path.join(viz_dir, 'temporal_analysis.png'))
            plt.close()

        # 3. Email Communication Network
        plt.figure(figsize=(15, 8))
        top_emails = dict(sorted(self.email_patterns.items(), key=lambda x: x[1], reverse=True)[:20])
        plt.bar(range(len(top_emails)), list(top_emails.values()))
        plt.xticks(range(len(top_emails)), list(top_emails.keys()), rotation=45, ha='right')
        plt.title('Top Email Correspondents')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'email_network.png'))
        plt.close()

        # 4. Sentiment Analysis
        if self.sentiment_scores:
            plt.figure(figsize=(15, 8))
            sns.histplot(self.sentiment_scores, bins=50)
            plt.title('Distribution of Sentiment Scores')
            plt.xlabel('Sentiment Score')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(viz_dir, 'sentiment_distribution.png'))
            plt.close()

        # 5. Location Heat Map (if matplotlib-basemap is available)
        if self.location_data:
            try:
                from mpl_toolkits.basemap import Basemap
                plt.figure(figsize=(15, 8))
                m = Basemap(projection='mill', llcrnrlat=-90, urcrnrlat=90,
                          llcrnrlon=-180, urcrnrlon=180, resolution='l')
                m.drawcoastlines()
                m.drawcountries()
                
                lats, lons = zip(*self.location_data)
                x, y = m(lons, lats)
                m.scatter(x, y, marker='o', color='red', alpha=0.5)
                
                plt.title('Location Data Heat Map')
                plt.savefig(os.path.join(viz_dir, 'location_heatmap.png'))
                plt.close()
            except ImportError:
                logging.warning("Basemap not available. Skipping location visualization.")

    def generate_report(self):
        """Generate a comprehensive analysis report."""
        report_path = os.path.join(self.output_directory, 'analysis_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Facebook Data Analysis Report\n\n")
            
            f.write("## Summary Statistics\n")
            f.write(f"- Total unique words analyzed: {len(self.word_counts)}\n")
            f.write(f"- Total timestamps found: {len(self.timestamps)}\n")
            f.write(f"- Total unique email addresses: {len(self.email_patterns)}\n")
            f.write(f"- Total location data points: {len(self.location_data)}\n\n")
            
            f.write("## Top Words\n")
            for word, count in self.word_counts.most_common(20):
                f.write(f"- {word}: {count}\n")
            
            f.write("\n## JSON Structure Analysis\n")
            for key, types in self.json_structure.items():
                f.write(f"- {key}: {', '.join(types)}\n")
            
            f.write("\n## Temporal Analysis\n")
            if self.timestamps:
                df_timestamps = pd.DataFrame(self.timestamps, columns=['timestamp'])
                yearly_counts = df_timestamps.groupby(df_timestamps['timestamp'].dt.year).size()
                f.write("Activity by year:\n")
                for year, count in yearly_counts.items():
                    f.write(f"- {year}: {count} events\n")
            
            f.write("\n## Sentiment Analysis\n")
            if self.sentiment_scores:
                f.write(f"- Average sentiment score: {np.mean(self.sentiment_scores):.2f}\n")
                f.write(f"- Median sentiment score: {np.median(self.sentiment_scores):.2f}\n")
                f.write(f"- Standard deviation: {np.std(self.sentiment_scores):.2f}\n")



def main():
    # Initialize and run the analyzer
    analyzer = FacebookDataAnalyzer(
        data_directory='/content/drive/MyDrive/MHHOctf/SeansGdata',
        output_directory='./FB_output2/'
    )
    
    analyzer.analyze_all_data()
    logging.info("Analysis complete! Check the output directory for results.")


if __name__ == "__main__":
    main()


