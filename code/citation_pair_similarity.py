"""
Patent Citation Pair Similarity Calculator

This script calculates Jaccard similarity between citation pairs (citing-cited patents)
based on their titles and abstracts.
"""

import os
import csv
import time
import argparse
from collections import defaultdict
import re
from EnglishStopWords import EnglishStopWords

class PatentCitationSimilarity:
    
    def __init__(self):
        """Initialize the similarity calculator"""
        self.sw = EnglishStopWords()  # Stopword list for preprocessing
        self.patents_data = {}  # Will store processed patent text data
    
    def get_token_pattern(self):
        """Define regex pattern to extract valid tokens from text"""
        return re.compile(r'\b[a-zA-Z0-9][-a-zA-Z0-9]*[a-zA-Z0-9]\b')
    
    def tokenize(self, text):
        """Split text into tokens and convert to lowercase"""
        if not text or not isinstance(text, str):
            return []
            
        tokens = []
        matcher = self.get_token_pattern().finditer(text)
        for match in matcher:
            tokens.append(match.group().lower())
        return tokens
    
    def preprocess_patent_text(self, title, abstract):
        """Create a clean set of tokens from patent title and abstract"""
        # Combine title and abstract
        text = f"{title} {abstract}" if title and abstract else (title or abstract or "")
        text = text.lower()
        
        # Tokenize and filter
        tokens = self.tokenize(text)
        clean_tokens = set()
        
        for token in tokens:
            # Remove stopwords, words formed only by numbers and words of only one character
            if (not self.sw.is_stop_word(token) and 
                len(token) > 1 and 
                not token.isdigit() and
                not re.match(r'[0-9]+(?:-[0-9]+)+$', token)):
                clean_tokens.add(token)
        
        return clean_tokens
    
    def load_patent_data(self, patents_file):
        """
        Load and preprocess patent data from CSV file
        
        Args:
            patents_file: CSV file with pnrn,title,abstract columns
        """
        print(f"Loading patent data from {patents_file}...")
        start_time = time.time()
        counter = 0
        
        with open(patents_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pnrn = row['pnrn'].strip()
                title = row.get('title_en', '').strip()
                abstract = row.get('abstract_en', '').strip()
                
                # Preprocess text into token set
                token_set = self.preprocess_patent_text(title, abstract)
                
                # Store only if we have valid tokens
                if token_set:
                    self.patents_data[pnrn] = token_set
                else:
                    print(f"Warning: No valid tokens for patent {pnrn}")
                
                counter += 1
                if counter % 100000 == 0:
                    print(f"Processed {counter} patents...")
        
        elapsed = time.time() - start_time
        print(f"Loaded {len(self.patents_data)} patents with valid text in {elapsed:.2f} seconds")
    
    def calculate_jaccard_similarity(self, set_a, set_b):
        """Calculate Jaccard similarity between two sets"""
        if not set_a or not set_b:
            return 0.0
            
        intersection = len(set_a.intersection(set_b))
        if intersection == 0:
            return 0.0
            
        union = len(set_a.union(set_b))
        return intersection / union
    
    def process_citation_pairs(self, citation_file, output_file):
        """
        Process citation pairs and calculate similarity for each pair
        
        Args:
            citation_file: CSV file with citing_ida,uniq_cited_id columns
            output_file: Output file to write results
        """
        print(f"Processing citation pairs from {citation_file}...")
        start_time = time.time()
        
        # Count statistics
        total_pairs = 0
        processed_pairs = 0
        missing_patents = set()
        
        with open(citation_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            # Write header
            f_out.write("citing_ida,uniq_cited_id,similarity\n")
            
            # Skip header in input file
            reader = csv.DictReader(f_in)
            
            for row in reader:
                total_pairs += 1
                citing_id = row['citing_ida'].strip()
                cited_id = row['uniq_cited_id'].strip()
                
                # Check if both patents exist in our data
                if citing_id in self.patents_data and cited_id in self.patents_data:
                    # Calculate Jaccard similarity
                    similarity = self.calculate_jaccard_similarity(
                        self.patents_data[citing_id], 
                        self.patents_data[cited_id]
                    )
                    
                    # Write result to output file
                    f_out.write(f"{citing_id},{cited_id},{similarity:.6f}\n")
                    processed_pairs += 1
                else:
                    # Track missing patents
                    if citing_id not in self.patents_data:
                        missing_patents.add(citing_id)
                    if cited_id not in self.patents_data:
                        missing_patents.add(cited_id)
                
                # Progress reporting
                if total_pairs % 10000 == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {total_pairs} citation pairs, "
                          f"successfully calculated {processed_pairs} similarities...")
        
        elapsed = time.time() - start_time
        print(f"Completed processing {total_pairs} citation pairs in {elapsed:.2f} seconds.")
        print(f"Successfully calculated similarity for {processed_pairs} pairs.")
        print(f"Number of missing patents: {len(missing_patents)}")
        
        # Write list of missing patents to file for reference
        if missing_patents:
            missing_file = f"{os.path.splitext(output_file)[0]}_missing_patents.txt"
            with open(missing_file, 'w', encoding='utf-8') as f:
                for pnrn in sorted(missing_patents):
                    f.write(f"{pnrn}\n")
            print(f"List of missing patents written to {missing_file}")

def main():
    parser = argparse.ArgumentParser(description='Calculate similarity between patent citation pairs')
    parser.add_argument('--patents', required=True, help='CSV file with patent data (pnrn,title,abstract)')
    parser.add_argument('--citations', required=True, help='CSV file with citation pairs (citing_ida,uniq_cited_id)')
    parser.add_argument('--output', required=True, help='Output file for similarity results')
    args = parser.parse_args()
    
    # Create directory for output file if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize and run the similarity calculator
    calculator = PatentCitationSimilarity()
    calculator.load_patent_data(args.patents)
    calculator.process_citation_pairs(args.citations, args.output)
    
    print("Patent citation similarity calculation completed!")

if __name__ == "__main__":
    main()