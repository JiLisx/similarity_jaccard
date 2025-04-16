"""
Patent Citation Pair Similarity Calculator

This script calculates Jaccard similarity between citation pairs (citing-cited patents)
based on their titles and abstracts, with multiprocessing support and TXT output.
"""

import os
import csv
import time
import argparse
from collections import defaultdict
import re
import multiprocessing as mp
from functools import partial
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
            patents_file: CSV file with pnrn,title_en,abstract_en columns
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
    
    def process_chunk(self, chunk, patent_data):
        """
        Process a chunk of citation pairs
        
        Args:
            chunk: List of (citing_id, cited_id) tuples
            patent_data: Dictionary of patent ID to token set
            
        Returns:
            List of (citing_id, cited_id, similarity) and set of missing patents
        """
        results = []
        missing_patents = set()
        
        for citing_id, cited_id in chunk:
            if citing_id in patent_data and cited_id in patent_data:
                similarity = self.calculate_jaccard_similarity(
                    patent_data[citing_id], 
                    patent_data[cited_id]
                )
                results.append((citing_id, cited_id, similarity))
            else:
                if citing_id not in patent_data:
                    missing_patents.add(citing_id)
                if cited_id not in patent_data:
                    missing_patents.add(cited_id)
                    
        return results, missing_patents
    
    def process_citation_pairs(self, citation_file, output_file, num_processes=None):
        """
        Process citation pairs and calculate similarity for each pair using multiprocessing
        
        Args:
            citation_file: CSV file with citing_ida,uniq_cited_id columns
            output_file: Output file to write results
            num_processes: Number of processes to use (default: CPU count)
        """
        print(f"Processing citation pairs from {citation_file}...")
        start_time = time.time()
        
        # Determine number of processes
        if num_processes is None:
            num_processes = mp.cpu_count()
        
        print(f"Using {num_processes} processes for parallel computation")
        
        # Read all citation pairs
        citation_pairs = []
        with open(citation_file, 'r', encoding='utf-8') as f_in:
            reader = csv.DictReader(f_in)
            for row in reader:
                citing_id = row['citing_ida'].strip()
                cited_id = row['uniq_cited_id'].strip()
                citation_pairs.append((citing_id, cited_id))
        
        total_pairs = len(citation_pairs)
        print(f"Found {total_pairs} citation pairs to process")
        
        # If just a few pairs, use single process
        if total_pairs < 100 or num_processes == 1:
            print("Using single process due to small number of pairs")
            results, missing_patents = self.process_chunk(citation_pairs, self.patents_data)
        else:
            # Split work into chunks for multiprocessing
            chunk_size = max(1, total_pairs // (num_processes * 2))
            chunks = [citation_pairs[i:i + chunk_size] for i in range(0, total_pairs, chunk_size)]
            print(f"Split work into {len(chunks)} chunks (chunk size: ~{chunk_size})")
            
            # Process chunks in parallel
            pool = mp.Pool(processes=num_processes)
            process_func = partial(self.process_chunk, patent_data=self.patents_data)
            chunk_results = pool.map(process_func, chunks)
            pool.close()
            pool.join()
            
            # Combine results
            results = []
            missing_patents = set()
            for chunk_result, chunk_missing in chunk_results:
                results.extend(chunk_result)
                missing_patents.update(chunk_missing)
        
        # Write results to output file (TXT format, space-separated)
        processed_pairs = len(results)
        print(f"Writing {processed_pairs} similarity results to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f_out:
            # Write header
            f_out.write("citing_ida,uniq_cited_id,similarity\n")
            
            # Write results sorted by citing_id then cited_id for consistency
            for citing_id, cited_id, similarity in sorted(results):
                f_out.write(f"{citing_id},{cited_id},{similarity:.6f}\n")
        
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
    parser.add_argument('--patents', required=True, help='CSV file with patent data (pnrn,title_en,abstract_en)')
    parser.add_argument('--citations', required=True, help='CSV file with citation pairs (citing_ida,uniq_cited_id)')
    parser.add_argument('--output', required=True, help='Output file for similarity results (TXT format)')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes to use (default: CPU count)')
    args = parser.parse_args()
    
    # Create directory for output file if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize and run the similarity calculator
    calculator = PatentCitationSimilarity()
    calculator.load_patent_data(args.patents)
    calculator.process_citation_pairs(args.citations, args.output, args.processes)
    
    print("Patent citation similarity calculation completed!")

if __name__ == "__main__":
    main()