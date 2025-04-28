"""
Patent Citation Pair Similarity Calculator - Streaming Version

This script calculates Jaccard similarity between citation pairs (citing-cited patents)
using a streaming approach that minimizes memory usage by processing one pair at a time.
"""

import os
import csv
import time
import argparse
import re
import multiprocessing as mp
from functools import partial
from EnglishStopWords import EnglishStopWords

class StreamingPatentSimilarity:
    
    def __init__(self):
        """Initialize the similarity calculator"""
        self.sw = EnglishStopWords()  # Stopword list for preprocessing
        self.patent_cache = {}  # Small LRU cache for recently used patents
        self.max_cache_size = 1000  # Maximum number of patents to keep in cache
    
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
    
    def create_patent_index(self, patents_file):
        """
        Create an index of patent IDs to their positions in the file for fast lookup
        
        Args:
            patents_file: CSV file with patent data
            
        Returns:
            Dictionary mapping patent IDs to file positions
        """
        print(f"Creating patent index for {patents_file}...")
        patent_positions = {}
        
        with open(patents_file, 'r', encoding='utf-8') as f:
            # Skip header
            header_pos = f.tell()
            header = f.readline()
            
            # Find column indices
            header_cols = header.strip().split(',')
            pnr_idx = header_cols.index('pnr') if 'pnr' in header_cols else 0
            
            # Record position of each patent
            while True:
                position = f.tell()
                line = f.readline()
                if not line:
                    break
                    
                # Parse just enough to get patent ID
                parts = line.split(',')
                if len(parts) > pnr_idx:
                    patent_id = parts[pnr_idx].strip()
                    patent_positions[patent_id] = position
        
        print(f"Indexed {len(patent_positions)} patents")
        return patent_positions, header_pos
    
    def get_patent_data(self, patent_id, patents_file, patent_positions, header_pos):
        """
        Retrieve patent data for a specific patent ID
        
        Args:
            patent_id: Patent ID to retrieve
            patents_file: CSV file with patent data
            patent_positions: Dictionary mapping patent IDs to file positions
            header_pos: Position of the header line in the file
            
        Returns:
            Set of tokens for the patent or None if not found
        """
        # Check cache first
        if patent_id in self.patent_cache:
            return self.patent_cache[patent_id]
        
        # If not in cache, check if we have position information
        if patent_id in patent_positions:
            with open(patents_file, 'r', encoding='utf-8') as f:
                # Read header to get column indices
                f.seek(header_pos)
                header = f.readline().strip().split(',')
                
                title_idx = header.index('title_en') if 'title_en' in header else 1
                abstract_idx = header.index('abstract_en') if 'abstract_en' in header else 2
                
                # Jump to the patent's position
                f.seek(patent_positions[patent_id])
                patent_line = f.readline()
                
                # Parse the CSV line
                parts = patent_line.split(',')
                if len(parts) > max(title_idx, abstract_idx):
                    title = parts[title_idx].strip() if title_idx < len(parts) else ""
                    
                    # Handle the case where the abstract might contain commas
                    if abstract_idx < len(parts):
                        # This is a simplification - real CSV parsing is more complex
                        # with quotes and escaped characters
                        abstract = ','.join(parts[abstract_idx:]).strip()
                        # Remove quotes if present
                        if abstract.startswith('"') and abstract.endswith('"'):
                            abstract = abstract[1:-1]
                    else:
                        abstract = ""
                    
                    # Preprocess text
                    token_set = self.preprocess_patent_text(title, abstract)
                    
                    # Update cache
                    self._update_cache(patent_id, token_set)
                    
                    return token_set
        
        # Patent not found
        return None
    
    def _update_cache(self, patent_id, token_set):
        """Update the patent cache, removing oldest entries if necessary"""
        if len(self.patent_cache) >= self.max_cache_size:
            # Remove a random item from cache
            # In a more sophisticated implementation, we'd use LRU strategy
            self.patent_cache.pop(next(iter(self.patent_cache)))
        
        self.patent_cache[patent_id] = token_set
    
    def calculate_jaccard_similarity(self, set_a, set_b):
        """Calculate Jaccard similarity between two sets"""
        if not set_a or not set_b:
            return 0.0
            
        intersection = len(set_a.intersection(set_b))
        if intersection == 0:
            return 0.0
            
        union = len(set_a.union(set_b))
        return intersection / union
    
    def process_citation_pairs(self, patents_file, citation_file, output_file, num_processes=None):
        """
        Process citation pairs and calculate similarity for each pair using streaming approach
        
        Args:
            patents_file: CSV file with pnr,title_en,abstract_en columns
            citation_file: CSV file with citing_pnr,cited_pnr columns
            output_file: Output file to write results
            num_processes: Number of processes to use for parallel processing
        """
        print(f"Processing citation pairs from {citation_file}...")
        start_time = time.time()
        
        # Determine number of processes
        if num_processes is None:
            num_processes = mp.cpu_count()
        
        print(f"Using {num_processes} processes for parallel computation")
        
        # Create patent index for fast lookup
        patent_positions, header_pos = self.create_patent_index(patents_file)
        
        # 检查断点续传
        processed_pairs_set = set()
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            # 读取已处理的引用对
            with open(output_file, 'r', encoding='utf-8') as f:
                next(f)  # 跳过标题行
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        processed_pairs_set.add((parts[0], parts[1]))
            
            print(f"Resuming from {len(processed_pairs_set)} previously processed pairs")
        
        # Set up counters
        total_pairs = 0
        processed_count = 0
        successful_pairs = 0
        missing_patents = set()
        
        # Process in chunks for better performance with multiprocessing
        chunk_size = 10000
        current_chunk = []
        
        # 确定是新建文件还是追加模式
        write_mode = 'a' if processed_pairs_set else 'w'
        
        # Open output file for streaming writes
        with open(output_file, write_mode, encoding='utf-8') as f_out:
            # 只有在新建文件时才写入标题
            if write_mode == 'w':
                f_out.write("citing_pnr,cited_pnr,similarity\n")
            
            # Read citation pairs
            with open(citation_file, 'r', encoding='utf-8') as f_in:
                reader = csv.DictReader(f_in)
                
                for row in reader:
                    citing_pnr = row['citing_pnr'].strip()
                    cited_pnr = row['cited_pnr'].strip()
                    total_pairs += 1
                    
                    # 检查是否已处理过这对引用
                    if (citing_pnr, cited_pnr) in processed_pairs_set:
                        continue
                    
                    current_chunk.append((citing_pnr, cited_pnr))
                    
                    # Process chunk when it reaches the desired size
                    if len(current_chunk) >= chunk_size:
                        if num_processes > 1 and len(current_chunk) > 100:
                            self._process_chunk_parallel(current_chunk, patents_file, patent_positions, 
                                                        header_pos, f_out, num_processes, missing_patents)
                        else:
                            self._process_chunk_sequential(current_chunk, patents_file, patent_positions, 
                                                          header_pos, f_out, missing_patents)
                        
                        # Update processed pairs count
                        processed_count += len(current_chunk)
                        successful_pairs += len(current_chunk) - len(missing_patents)
                        current_chunk = []
                        
                        # Print progress
                        print(f"Processed {processed_count}/{total_pairs} citation pairs...")
                
                # Process any remaining pairs
                if current_chunk:
                    if num_processes > 1 and len(current_chunk) > 100:
                        self._process_chunk_parallel(current_chunk, patents_file, patent_positions, 
                                                    header_pos, f_out, num_processes, missing_patents)
                    else:
                        self._process_chunk_sequential(current_chunk, patents_file, patent_positions, 
                                                      header_pos, f_out, missing_patents)
                    
                    processed_count += len(current_chunk)
                    successful_pairs += len(current_chunk) - len(missing_patents)
        
        elapsed = time.time() - start_time
        print(f"Completed processing {total_pairs} citation pairs in {elapsed:.2f} seconds.")
        print(f"Successfully calculated similarity for {successful_pairs} pairs.")
        print(f"Number of missing patents: {len(missing_patents)}")
        
        # Write list of missing patents to file for reference
        if missing_patents:
            missing_file = f"{os.path.splitext(output_file)[0]}_missing_patents.txt"
            with open(missing_file, 'w', encoding='utf-8') as f:
                for pnr in sorted(missing_patents):
                    f.write(f"{pnr}\n")
            print(f"List of missing patents written to {missing_file}")

    def _process_chunk_sequential(self, chunk, patents_file, patent_positions, header_pos, output_file, missing_patents):
        """Process a chunk of citation pairs sequentially"""
        for citing_pnr, cited_pnr in chunk:
            # Get patent data
            citing_tokens = self.get_patent_data(citing_pnr, patents_file, patent_positions, header_pos)
            cited_tokens = self.get_patent_data(cited_pnr, patents_file, patent_positions, header_pos)
            
            # Calculate similarity if both patents exist
            if citing_tokens and cited_tokens:
                similarity = self.calculate_jaccard_similarity(citing_tokens, cited_tokens)
                output_file.write(f"{citing_pnr},{cited_pnr},{similarity:.6f}\n")
            else:
                # Record missing patents
                if not citing_tokens:
                    missing_patents.add(citing_pnr)
                if not cited_tokens:
                    missing_patents.add(cited_pnr)
    
    def _process_single_pair(self, pair_data):
        """
        Process a single citation pair, used by the parallel processor
        
        Args:
            pair_data: Tuple of (citing_pnr, cited_pnr, patents_file, patent_positions, header_pos)
            
        Returns:
            Tuple of (citing_pnr, cited_pnr, similarity) or None if a patent is missing
        """
        citing_pnr, cited_pnr, patents_file, patent_positions, header_pos = pair_data
        
        # Get patent data
        citing_tokens = self.get_patent_data(citing_pnr, patents_file, patent_positions, header_pos)
        cited_tokens = self.get_patent_data(cited_pnr, patents_file, patent_positions, header_pos)
        
        # Calculate similarity if both patents exist
        if citing_tokens and cited_tokens:
            similarity = self.calculate_jaccard_similarity(citing_tokens, cited_tokens)
            return (citing_pnr, cited_pnr, similarity)
        else:
            # Return information about missing patents
            missing = []
            if not citing_tokens:
                missing.append(citing_pnr)
            if not cited_tokens:
                missing.append(cited_pnr)
            return None, missing
    
    def _process_chunk_parallel(self, chunk, patents_file, patent_positions, header_pos, 
                               output_file, num_processes, missing_patents):
        """Process a chunk of citation pairs in parallel"""
        # Prepare data for parallel processing
        pair_data = [(citing_pnr, cited_pnr, patents_file, patent_positions, header_pos) 
                      for citing_pnr, cited_pnr in chunk]
        
        # Process pairs in parallel
        pool = mp.Pool(processes=num_processes)
        results = pool.map(self._process_single_pair, pair_data)
        pool.close()
        pool.join()
        
        # Write results
        for result in results:
            if result[0] is not None:
                # Valid similarity result
                citing_pnr, cited_pnr, similarity = result
                output_file.write(f"{citing_pnr},{cited_pnr},{similarity:.6f}\n")
            else:
                # Missing patents
                for patent_id in result[1]:
                    missing_patents.add(patent_id)


def main():
    parser = argparse.ArgumentParser(description='Calculate similarity between patent citation pairs using streaming approach')
    parser.add_argument('--patents', required=True, help='CSV file with patent data (pnr,title_en,abstract_en)')
    parser.add_argument('--citations', required=True, help='CSV file with citation pairs (citing_pnr,cited_pnr)')
    parser.add_argument('--output', required=True, help='Output file for similarity results')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes to use (default: CPU count)')
    parser.add_argument('--cache-size', type=int, default=1000, help='Maximum number of patents to keep in cache (default: 1000)')
    args = parser.parse_args()
    
    # Create directory for output file if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize the similarity calculator
    calculator = StreamingPatentSimilarity()
    calculator.max_cache_size = args.cache_size
    
    # Process citation pairs
    calculator.process_citation_pairs(args.patents, args.citations, args.output, args.processes)
    
    print("Patent citation similarity calculation completed!")

if __name__ == "__main__":
    main()