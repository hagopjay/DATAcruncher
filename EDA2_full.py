# !pip install nltk wordcloud scikit-learn tqdm seaborn pandas numpy matplotlib


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
from typing import Dict, List, Set, Any
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler('google_data_analysis.log'),
                           logging.StreamHandler()])

class GoogleDataAnalyzer:
    def __init__(self, zip_directory: str, output_directory: str):
        self.zip_directory = zip_directory
        self.output_directory = output_directory
        self.media_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.mp4', 
                               '.mov', '.m4v', '.mp3', '.heic', 'hvac'}
        
        # Initialize data structures
        self.word_counts = Counter()
        self.text_data = []
        self.json_structure = defaultdict(set)
        self.timestamps = []
        self.email_patterns = defaultdict(int)
        self.location_data = []
        self.sentiment_scores = []
        
        # Download required NLTK data
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('vader_lexicon')
        
        # Create output directory
        os.makedirs(output_directory, exist_ok=True)

    def process_zip_files(self):
        """Process all ZIP files in the directory."""
        zip_files = [f for f in os.listdir(self.zip_directory) if f.endswith('.zip')]
        
        for zip_file in tqdm(zip_files, desc="Processing ZIP files"):
            zip_path = os.path.join(self.zip_directory, zip_file)
            self.process_single_zip(zip_path)

    def process_single_zip(self, zip_path: str):
        """Process a single ZIP file."""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if file_info.filename.endswith(('.json', '.jsonl', '.txt')):
                    try:
                        with zip_ref.open(file_info.filename) as file:
                            content = file.read().decode('utf-8')
                            
                            if file_info.filename.endswith(('.json', '.jsonl')):
                                self.analyze_json_content(content, file_info.filename)
                            
                            self.analyze_text_content(content)
                            self.extract_timestamps(content)
                            self.analyze_email_patterns(content)
                            self.extract_location_data(content)
                            self.analyze_sentiment(content)
                            
                    except Exception as e:
                        logging.error(f"Error processing {file_info.filename}: {str(e)}")

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
            f.write("# Google Data Analysis Report\n\n")
            
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
    analyzer = GoogleDataAnalyzer(
        zip_directory='/path/to/your/google/data/',
        output_directory='/path/to/output/'
    )
    
    analyzer.process_zip_files()
    analyzer.generate_visualizations()
    analyzer.generate_report()
    
    logging.info("Analysis complete! Check the output directory for results.")

if __name__ == "__main__":
    main()
