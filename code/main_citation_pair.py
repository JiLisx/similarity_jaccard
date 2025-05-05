"""
Patent Citation Pair Similarity
Author: Ji Li
Date: May 3, 2025
"""

import os
import csv
import time
import argparse
import re
import multiprocessing as mp
import gc


# Import required components from the existing codebase
from Stage01PreprocessData import Stage01PreprocessData
from EnglishStopWords import EnglishStopWords

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def load_vocabulary(vocabulary_path):
    """
    Load the shared vocabulary using the same method as in main_script.py
    """
    print(f"Loading shared vocabulary from {vocabulary_path}...")
    vocabulary = {}
    
    # Use the same preprocessing class for consistency
    ppd = Stage01PreprocessData()
    ppd.read_vocabulary(vocabulary_path, vocabulary)
    
    print(f"Loaded {len(vocabulary)} terms from shared vocabulary")
    return vocabulary

def create_patent_index(patents_file):
    """
    Create an index of patent IDs to their positions in the file for fast lookup
    """
    print(f"Creating patent index for {patents_file}...")
    patent_positions = {}
    
    with open(patents_file, 'r', encoding='utf-8') as f:
        # Skip header
        header_pos = f.tell()
        header = f.readline()
        
        # Find column indices using csv reader for proper handling
        csv_reader = csv.reader([header])
        header_cols = next(csv_reader)
        pnr_idx = header_cols.index('pnr') if 'pnr' in header_cols else 0
        
        # Record position of each patent
        line_count = 0
        while True:
            position = f.tell()
            line = f.readline()
            if not line:
                break
                
            # Parse just enough to get patent ID using csv reader
            csv_line_reader = csv.reader([line])
            parts = next(csv_line_reader)
            if len(parts) > pnr_idx:
                patent_id = parts[pnr_idx].strip()
                patent_positions[patent_id] = position
            
            line_count += 1
            if line_count % 1000000 == 0:
                print(f"  Indexed {line_count} patents...")
    
    print(f"Indexed {len(patent_positions)} patents")
    return patent_positions, header_pos

def get_patent_data(patent_id, patents_file, patent_positions, header_pos, vocabulary, patent_cache, sw, ppd):
    """
    Retrieve and preprocess patent data for a specific patent ID
    """
    # Check cache first
    if patent_id in patent_cache:
        return patent_cache[patent_id]
    
    # If not in cache, check if we have position information
    if patent_id in patent_positions:
        with open(patents_file, 'r', encoding='utf-8') as f:
            # Read header to get column indices
            f.seek(header_pos)
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            
            title_idx = header.index('title_en') if 'title_en' in header else 2
            abstract_idx = header.index('abstract_en') if 'abstract_en' in header else 3
            
            # Jump to the patent's position
            f.seek(patent_positions[patent_id])
            line = f.readline()
            
            # Parse the CSV line properly
            csv_line_reader = csv.reader([line])
            parts = next(csv_line_reader)
            
            if len(parts) > max(title_idx, abstract_idx):
                # Extract title and abstract using the indices
                if title_idx < len(parts) and abstract_idx < len(parts):
                    title = parts[title_idx].strip()
                    abstract = parts[abstract_idx].strip()
                    
                    # Check if title and abstract are not empty
                    if not title or not abstract:
                        return None
                else:
                    return None 
                
                # Combine title and abstract - using exactly the same approach as Stage01PreprocessData
                text = f"{title} {abstract}".lower()
                
                # Use the original tokenization method from Stage01PreprocessData
                tokens = ppd.tokenize(text)
                
                # Filter exactly as in Stage01PreprocessData.create_bag_of_words
                filtered_tokens = set()
                for token in tokens:
                    # Apply the same filtering rules as in Stage01PreprocessData
                    if (token in vocabulary and
                        not sw.is_stop_word(token) and
                        len(token) > 1 and
                        not token.isdigit() and  # Filter pure numbers
                        not re.match(r'[0-9]+(?:-[0-9]+)+$', token)):
                        filtered_tokens.add(token)
                
                # Update cache with a copy of the set
                if len(patent_cache) >= 10000:  # Fixed cache size
                    # Remove least recently used item (first in the cache)
                    patent_cache.pop(next(iter(patent_cache)))
                patent_cache[patent_id] = filtered_tokens.copy()
                
                return filtered_tokens
    
    # Patent not found
    return None

def calculate_jaccard_similarity(set_a, set_b):
    """
    Calculate Jaccard similarity between two sets with rounding to 6 decimal places,
    using exactly the same formula as in main_citation_control.py
    """
    # Return 0 for empty sets - consistent with other implementations
    if not set_a or not set_b:
        return 0.0
        
    # Calculate intersection
    intersection = len(set_a.intersection(set_b))
        
    # Calculate union using set operations to ensure consistency
    union = len(set_a.union(set_b))
    
    # Compute similarity and round to 6 decimal places - exactly like Stage05ComputeSimilarity
    similarity = intersection / union
    return round(similarity, 6)

# Cache to share patent data between worker processes (shared dictionary)
# Note: In Python multiprocessing, each process gets its own copy of global variables
# So we'll create a manager to handle shared data if needed
def initialize_worker(patents_file_arg, patent_positions_arg, header_pos_arg, vocabulary_arg):
    """Initialize worker with shared data"""
    global patents_file, patent_positions, header_pos, vocabulary, patent_cache, sw, ppd
    patents_file = patents_file_arg
    patent_positions = patent_positions_arg
    header_pos = header_pos_arg
    vocabulary = vocabulary_arg
    patent_cache = {}  # Each worker maintains its own cache
    sw = EnglishStopWords()  # Create stopwords instance
    ppd = Stage01PreprocessData()  # Create preprocessor instance

def process_citation_pair(pair_data):
    """Process a single citation pair"""
    citing_pnr, cited_pnr = pair_data
    
    # Access the global variables set in initialize_worker
    global patents_file, patent_positions, header_pos, vocabulary, patent_cache, sw, ppd
    
    # Get patent data
    citing_tokens = get_patent_data(citing_pnr, patents_file, patent_positions, 
                                   header_pos, vocabulary, patent_cache, sw, ppd)
    cited_tokens = get_patent_data(cited_pnr, patents_file, patent_positions, 
                                  header_pos, vocabulary, patent_cache, sw, ppd)
    
    # Calculate similarity if both patents exist
    if citing_tokens is not None and cited_tokens is not None:
        similarity = calculate_jaccard_similarity(citing_tokens, cited_tokens)
        return (citing_pnr, cited_pnr, similarity, None)
    else:
        # Return information about missing patents
        missing = []
        if citing_tokens is None:
            missing.append(citing_pnr)
        if cited_tokens is None:
            missing.append(cited_pnr)
        return (citing_pnr, cited_pnr, None, missing)

def process_citation_pairs(patents_file, citation_file, output_file, vocabulary_path, num_processes=None):
    """
    Process citation pairs and calculate similarity using improved parallel processing
    with better memory management and progress tracking
    """
    print(f"Processing citation pairs from {citation_file}...")
    print(f"Using {num_processes} processes for parallel calculation" if num_processes else "Using all available CPU cores")
    start_time = time.time()
    
    # Load vocabulary - using the exact same method as in main_script.py
    vocabulary = load_vocabulary(vocabulary_path)
    
    # Create patent index for fast lookup
    patent_positions, header_pos = create_patent_index(patents_file)
    
    # Count total number of citation pairs for progress reporting
    total_pairs_count = 0
    with open(citation_file, 'r', encoding='utf-8') as f:
        # Skip header
        next(f)
        # Count lines
        for _ in f:
            total_pairs_count += 1
    
    print(f"Total citation pairs to process: {total_pairs_count}")
    
    # Check for resuming from previous run
    processed_pairs = set()
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("citing_pnr"):  # Skip header
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    processed_pairs.add((parts[0], parts[1]))
        
        print(f"Resuming from {len(processed_pairs)} previously processed pairs")
    
    # Determine write mode
    write_mode = 'a' if processed_pairs else 'w'
    
    # Create output and missing patents files
    missing_patents_file = f"{os.path.splitext(output_file)[0]}_missing_patents.txt"
    with open(output_file, write_mode, encoding='utf-8') as f_out, \
         open(missing_patents_file, 'w', encoding='utf-8') as f_missing:
        
        # Write headers
        if write_mode == 'w':
            f_out.write("citing_pnr cited_pnr similarity\n")
        f_missing.write("Missing patents\n")
        
        # Track progress
        total_pairs = 0
        successful_pairs = 0
        missing_patents_count = 0
        last_progress_time = time.time()
        batch_size = 50000  # Larger batch size for better efficiency
        
        # Create citation pair batches
        citation_pairs = []
        with open(citation_file, 'r', encoding='utf-8') as f_in:
            csv_reader = csv.reader(f_in)
            header = next(csv_reader)
            
            # Find column indices
            citing_idx = header.index('citing_pnr') if 'citing_pnr' in header else 0
            cited_idx = header.index('cited_pnr') if 'cited_pnr' in header else 1
            
            for row in csv_reader:
                if len(row) <= max(citing_idx, cited_idx):
                    continue
                
                citing_pnr = row[citing_idx].strip()
                cited_pnr = row[cited_idx].strip()
                total_pairs += 1
                
                # Skip already processed pairs
                if (citing_pnr, cited_pnr) in processed_pairs:
                    continue
                
                # Add to batch
                citation_pairs.append((citing_pnr, cited_pnr))
                
                # Process batch when it reaches the desired size
                if len(citation_pairs) >= batch_size:
                    # Create a process pool with initialized workers
                    with mp.Pool(
                        processes=num_processes,
                        initializer=initialize_worker,
                        initargs=(patents_file, patent_positions, header_pos, vocabulary)
                    ) as pool:
                        # Process batch in parallel
                        for result in pool.map(process_citation_pair, citation_pairs):
                            citing_pnr, cited_pnr, similarity, missing = result
                            
                            if similarity is not None:
                                f_out.write(f"{citing_pnr} {cited_pnr} {similarity}\n")
                                successful_pairs += 1
                            else:
                                # Record missing patents
                                for patent_id in missing:
                                    f_missing.write(f"{patent_id}\n")
                                    missing_patents_count += 1
                    
                    # Clear batch and force garbage collection
                    citation_pairs = []
                    gc.collect()
                    
                    # Print progress
                    current_time = time.time()
                    if current_time - last_progress_time >= 5:
                        elapsed = current_time - start_time
                        progress = (total_pairs / total_pairs_count) * 100
                        pairs_per_sec = total_pairs / elapsed if elapsed > 0 else 0
                        estimated_remaining = (total_pairs_count - total_pairs) / pairs_per_sec if pairs_per_sec > 0 else 0
                        
                        print(f"Progress: {total_pairs}/{total_pairs_count} pairs ({progress:.2f}%)")
                        print(f"Speed: {pairs_per_sec:.1f} pairs/sec, ETA: {estimated_remaining/60:.1f} min")
                        print(f"Successful: {successful_pairs}, Missing patents: {missing_patents_count}")
                        print("-" * 70)
                        
                        last_progress_time = current_time
            
            # Process remaining pairs
            if citation_pairs:
                with mp.Pool(
                    processes=num_processes,
                    initializer=initialize_worker,
                    initargs=(patents_file, patent_positions, header_pos, vocabulary)
                ) as pool:
                    for result in pool.map(process_citation_pair, citation_pairs):
                        citing_pnr, cited_pnr, similarity, missing = result
                        
                        if similarity is not None:
                            f_out.write(f"{citing_pnr} {cited_pnr} {similarity}\n")
                            successful_pairs += 1
                        else:
                            # Record missing patents
                            for patent_id in missing:
                                f_missing.write(f"{patent_id}\n")
                                missing_patents_count += 1
    
    # Report results
    elapsed_time = time.time() - start_time
    
    print("\nResults:")
    print(f"- Total processed citation pairs: {total_pairs}")
    print(f"- Successfully calculated similarities: {successful_pairs}")
    print(f"- Missing patents: {missing_patents_count}")
    print(f"- Results written to: {output_file}")
    print(f"- Missing patents written to: {missing_patents_file}")
    print(f"- Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")


def main():
    parser = argparse.ArgumentParser(description='Calculate similarity between patent citation pairs with optimized multiprocessing')
    parser.add_argument('--patents', required=True, help='CSV file with patent data (pnr,title_en,abstract_en)')
    parser.add_argument('--citations', required=True, help='CSV file with citation pairs (citing_pnr,cited_pnr)')
    parser.add_argument('--output', required=True, help='Output file for similarity results (TXT format)')
    parser.add_argument('--dir', required=True, help='Working directory for the pipeline (same as used for main_script.py)')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes to use (default: CPU count)')
    args = parser.parse_args()
    
    # Define the vocabulary path relative to the main directory
    vocabulary_path = os.path.join(args.dir, "vocabulary_raw.txt")
    
    # If vocabulary not found in the specified directory, try parent directory
    if not os.path.exists(vocabulary_path):
        vocabulary_path = os.path.join(args.dir, "../data/vocabulary_raw.txt")
    
    # Verify that vocabulary file exists
    if not os.path.exists(vocabulary_path):
        print(f"ERROR: Vocabulary file not found in {args.dir} or ../data/")
        print("Please run main_script.py first to generate the vocabulary.")
        return
    
    # Create directory for output file if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        ensure_dir_exists(output_dir)
    
    # Process citation pairs with shared vocabulary and consistent preprocessing
    process_citation_pairs(
        patents_file=args.patents, 
        citation_file=args.citations, 
        output_file=args.output, 
        vocabulary_path=vocabulary_path,
        num_processes=args.processes
    )
    
if __name__ == "__main__":
    main()