"""
Patent Citation Similarity for Patent Control 
Author: Ji Li 
Date: May 3, 2025
"""
import os
import csv
import time
import argparse
import multiprocessing as mp
from collections import defaultdict
import heapq
import gc
import uuid

# Import required components from the existing codebase
from Stage01PreprocessData import Stage01PreprocessData
from Stage02CodifyIdxPatents import Stage02CodifyIdxPatents
from Stage03IndexPatents import Stage03IndexPatents

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def run_stage1(main_dir, patents_file):
    """Preprocess patent data (reusing existing vocabulary)"""
    print("\n" + "="*80)
    print("STAGE 1: Preprocessing Patent Data (with existing vocabulary)")
    print("="*80)
    
    # Define output files
    f_terms = os.path.join(main_dir, "patents_terms.txt")
    f_years = os.path.join(main_dir, "patents_years.txt")
    f_patents_idxs = os.path.join(main_dir, "patents_numbers.txt")
    f_vocabulary = os.path.join(main_dir, "../data/vocabulary_raw.txt")
    
    # Check if already preprocessed
    if all(os.path.exists(f) and os.path.getsize(f) > 0 for f in [f_terms, f_years, f_patents_idxs]):
        print("Stage 1 output files already exist, skipping...")
        return True
    
    # Check if vocabulary exists
    if not os.path.exists(f_vocabulary) or os.path.getsize(f_vocabulary) == 0:
        print(f"ERROR: Vocabulary file {f_vocabulary} not found or empty.")
        return False
    
    # Create bag-of-words from patent data
    ppd = Stage01PreprocessData()
    f_data = patents_file
    f_terms_raw = os.path.join(main_dir, "patents_terms_raw.txt")
    
    if not os.path.exists(f_data):
        print(f"ERROR: Input file {f_data} not found.")
        return False
    
    print("Creating bag-of-words file...")
    ppd.create_bag_of_words(f_data, f_terms_raw)

    # SKIP vocabulary extraction - use existing vocabulary
    print("Using existing vocabulary from main_script.py...")
    
    print("Cleaning the bag-of-words file...")
    vocabulary = {}
    ppd.read_vocabulary(f_vocabulary, vocabulary)
    ppd.clean_patents(f_terms_raw, f_terms, f_years, f_patents_idxs, vocabulary)
    
    return True


def run_stages_2_to_4(main_dir):
    """Run preprocessing stages 2-4 (codification, indexing, splitting by year)"""
    # Run Stage 2: Codify indices
    print("\n" + "="*80)
    print("STAGE 2: Codifying Patent and Vocabulary Indices")
    print("="*80)
    
    f_vocabulary = os.path.join(main_dir, "vocabulary.txt")
    f_patents_idxs = os.path.join(main_dir, "patents_idxs.txt")
    
    if all(os.path.exists(f) and os.path.getsize(f) > 0 for f in [f_vocabulary, f_patents_idxs]):
        print("Stage 2 output files already exist, skipping...")
    else:
        cip = Stage02CodifyIdxPatents()
        
        # Codify vocabulary
        f_vocab_raw = os.path.join(main_dir, "../data/vocabulary_raw.txt")
        if not os.path.exists(f_vocab_raw):
            print(f"ERROR: Input file {f_vocab_raw} not found.")
            return False
        
        print("Codifying vocabulary...")
        with open(f_vocab_raw, 'r', encoding='utf-8') as br_content, \
             open(f_vocabulary, 'w', encoding='utf-8') as pw_indexed:
            
            n = 0
            for line in br_content:
                code = cip.convert_to_code(n)
                pw_indexed.write(f"{code} {line.strip()}\n")
                n += 1
                if n % 10000 == 0:
                    print(f"\tProcessed = {n} words")
        
        # Codify patent numbers
        f_patents_num = os.path.join(main_dir, "patents_numbers.txt")
        if not os.path.exists(f_patents_num):
            print(f"ERROR: Input file {f_patents_num} not found.")
            return False
        
        print("Codifying patent numbers...")
        with open(f_patents_num, 'r', encoding='utf-8') as br_content, \
             open(f_patents_idxs, 'w', encoding='utf-8') as pw_indexed:
            
            n = 0
            for line in br_content:
                code = cip.convert_to_code(n)
                pw_indexed.write(f"{code} {line.strip()}\n")
                n += 1
                if n % 100000 == 0:
                    print(f"\tProcessed = {n} patents")
    
    # Run Stage 3: Index patents
    print("\n" + "="*80)
    print("STAGE 3: Indexing Patents")
    print("="*80)
    
    f_indexed = os.path.join(main_dir, "patents_indexed.txt")
    
    if os.path.exists(f_indexed) and os.path.getsize(f_indexed) > 0:
        print("Stage 3 output files already exist, skipping...")
    else:
        ip = Stage03IndexPatents()
        
        f_clean = os.path.join(main_dir, "patents_terms.txt")
        
        if not all(os.path.exists(f) for f in [f_clean, f_patents_idxs, f_vocabulary]):
            print("ERROR: Required input files for Stage 3 not found.")
            return False
        
        vocabulary = {}
        patents_idxs = {}
        
        print("Loading codified vocabulary...")
        ip.read_indexes(f_vocabulary, vocabulary)
        
        print("Loading codified patent numbers...")
        ip.read_indexes(f_patents_idxs, patents_idxs)
        
        print("Indexing patent data...")
        ip.index_patents(f_clean, f_indexed, vocabulary, patents_idxs)
    
    # Skip Stage 4 (splitting by year only) as we'll use year+type split instead
    print("\n" + "="*80)
    print("STAGE 4: Split by Year+Type)")
    print("="*80)

    # We still need to check if the years file exists for later use
    f_years = os.path.join(main_dir, "patents_years.txt")
    if not os.path.exists(f_years):
        print("ERROR: Required years file not found.")
        return False

    print("Stage 4 skipped - will use year+type split directly")
    
    return True


def create_patent_type_index(patents_file, main_dir):
    """Create an index mapping patent IDs to their types and split files by year and type."""
    print("\n" + "="*80)
    print("EXTENSION: Creating Patent Type Index and Year-Type Split")
    print("="*80)
    
    # Define files
    patent_type_map_file = os.path.join(main_dir, "patent_id_to_type.txt")
    years_types_dir = os.path.join(main_dir, "years_types")
    
    # Check if already done
    if os.path.exists(patent_type_map_file) and os.path.exists(years_types_dir) and len(os.listdir(years_types_dir)) > 0:
        print("Patent type index and year-type split already exist, loading...")
        
        # Load patent ID to type mapping
        patent_to_type = {}
        patent_to_year = {}
        
        with open(patent_type_map_file, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            for parts in csv_reader:
                if len(parts) >= 3:
                    patent_id, patent_type, year = parts
                    patent_to_type[patent_id] = patent_type.lower()
                    patent_to_year[patent_id] = year
        
        # Load codified to original patent ID mapping
        codified_to_original = {}
        original_to_codified = {}
        patents_idxs_file = os.path.join(main_dir, "patents_idxs.txt")
        
        with open(patents_idxs_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    codified, original = parts
                    codified_to_original[codified] = original
                    original_to_codified[original] = codified
        
        print(f"Loaded type information for {len(patent_to_type)} patents")
        return patent_to_type, patent_to_year, codified_to_original, original_to_codified
    
    # If not done already, create the patent type index and split by year and type
    ensure_dir_exists(years_types_dir)
    
    # Find relevant columns in the patents file
    with open(patents_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        
        # Find column indices
        pnr_idx = header.index('pnr') if 'pnr' in header else 0
        year_idx = header.index('year') if 'year' in header else 1
        type_idx = header.index('patent_type') if 'patent_type' in header else -1
        
        if type_idx == -1:
            print("ERROR: Patent type column not found in the CSV header.")
            return None, None, None, None
    
    # Load the patent years file
    f_years = os.path.join(main_dir, "patents_years.txt")
    years_list = []
    
    with open(f_years, 'r', encoding='utf-8') as f:
        for line in f:
            years_list.append(line.strip())
    
    # Load the patents numbers file
    patent_to_type = {}
    patent_to_year = {}
    patent_numbers_file = os.path.join(main_dir, "patents_numbers.txt")
    original_patents = []
    
    with open(patent_numbers_file, 'r', encoding='utf-8') as f:
        for line in f:
            original_patents.append(line.strip())
    
    # Read patent types from the original file
    print("Reading patent types from the original file...")
    
    total_patents = len(original_patents)
    types_assigned = 0
    
    with open(patents_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        
        for line_idx, parts in enumerate(csv_reader):
            if line_idx >= total_patents:
                break
                
            if line_idx % 1000000 == 0:
                print(f"Processed {line_idx} patents...")
            
            if len(parts) > max(pnr_idx, type_idx, year_idx):
                patent_id = parts[pnr_idx].strip()
                patent_type = parts[type_idx].strip().lower()
                year = parts[year_idx].strip()
                
                if patent_id and patent_type and year:
                    patent_to_type[patent_id] = patent_type
                    patent_to_year[patent_id] = year
                    types_assigned += 1
    
    print(f"Assigned types to {types_assigned} out of {total_patents} patents")
    
    # Save patent ID to type mapping for future use
    with open(patent_type_map_file, 'w', encoding='utf-8') as f:
        for patent_id, patent_type in patent_to_type.items():
            if patent_id in patent_to_year:
                f.write(f"{patent_id},{patent_type},{patent_to_year[patent_id]}\n")
    
    # Load codified to original mapping and vice versa
    codified_to_original = {}
    original_to_codified = {}
    f_patents_idxs = os.path.join(main_dir, "patents_idxs.txt")
    
    with open(f_patents_idxs, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                codified, original = parts
                codified_to_original[codified] = original
                original_to_codified[original] = codified
    
    # Split patents by year and type
    print("Splitting patent data by year and type...")
    f_indexed = os.path.join(main_dir, "patents_indexed.txt")
    year_type_files = {}
    
    with open(f_indexed, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if line_idx % 1000000 == 0:
                print(f"Processed {line_idx} patents for year-type split...")
            
            parts = line.strip().split(' ', 1)
            if len(parts) >= 1:
                codified_id = parts[0]
                if codified_id in codified_to_original:
                    original_id = codified_to_original[codified_id]
                    
                    if original_id in patent_to_year and original_id in patent_to_type:
                        year = patent_to_year[original_id]
                        patent_type = patent_to_type[original_id]
                        
                        year_type_key = f"{year}_{patent_type}"
                        if year_type_key not in year_type_files:
                            filename = os.path.join(years_types_dir, f"patents_{year_type_key}.txt")
                            year_type_files[year_type_key] = open(filename, 'w', encoding='utf-8')
                        
                        year_type_files[year_type_key].write(line)
    
    # Close all file handles
    for file_handle in year_type_files.values():
        file_handle.close()
    
    print(f"Created {len(year_type_files)} year-type specific files")
    return patent_to_type, patent_to_year, codified_to_original, original_to_codified


def load_cited_patents(citations_file):
    """Load cited patents from the citations file"""
    print(f"Loading cited patents from {citations_file}...")
    cited_patents = []
    
    with open(citations_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        
        # Check for required columns
        if 'cited_pnr' not in header and 'cited_patent' not in header:
            print("ERROR: Required 'cited_pnr' or 'cited_patent' column not found in citations file")
            return None
        
        if 'cited_patent_type' not in header and 'cited_type' not in header:
            print("ERROR: Required type column not found in citations file")
            print("Citations CSV must include a 'cited_patent_type' or 'cited_type' column")
            return None
        
        # Determine column names
        cited_pnr_col = 'cited_pnr' if 'cited_pnr' in header else 'cited_patent'
        cited_type_col = 'cited_patent_type' if 'cited_patent_type' in header else 'cited_type'
        
        # Look for year column
        year_col = None
        for possible_name in ['cited_year', 'year']:
            if possible_name in header:
                year_col = possible_name
                break
        
        for row in reader:
            cited_pnr = row.get(cited_pnr_col, '').strip()
            cited_type = row.get(cited_type_col, '').strip().lower()
            cited_year = row.get(year_col, '').strip() if year_col else ''
            
            if cited_pnr and cited_type:
                cited_tuple = (cited_pnr, cited_type, cited_year)
                if cited_tuple not in cited_patents:
                    cited_patents.append(cited_tuple)
    
    print(f"Loaded {len(cited_patents)} unique cited patents")
    return cited_patents


def get_processed_patents(output_file):
    """Load already processed patents from output file to support resuming"""
    processed = set()
    
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                # Skip header if present
                first_line = f.readline()
                if not first_line.startswith("cited_pnr"):
                    f.seek(0)
                
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        cited_pnr = parts[0]
                        processed.add(cited_pnr)  
        except Exception as e:
            print(f"Warning: Error reading existing output file: {e}")
            return set()
    
    return processed

def calculate_jaccard_similarity(set_a, set_b):
    """Calculate Jaccard similarity between two sets"""
    if not set_a or not set_b:
        return 0.0
    
    intersection = len(set_a.intersection(set_b))
    if intersection == 0:
        return 0.0
    
    union = len(set_a.union(set_b))
    similarity = intersection / union
    return round(similarity, 6)


def init_worker(main_dir, codified_to_original_arg, original_to_codified_arg, _):
    """Initialize worker process with shared data"""
    global _main_dir, codified_to_original, original_to_codified
    _main_dir = main_dir
    codified_to_original = codified_to_original_arg
    original_to_codified = original_to_codified_arg
    worker_id = f"worker_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    print(f"Worker {worker_id} initialized")
    return worker_id


def find_similar_patents_direct_top_n(cited_patents, main_dir, patent_to_year, patent_to_type, 
                                 original_to_codified, codified_to_original, output_file, 
                                 top_n=3, num_processes=None, batch_size=1000):
    """Find similar patents for cited patents with direct Top-N results return"""
    print(f"Finding top {top_n} similar patents for {len(cited_patents)} cited patents (direct Top-N return)...")
    
    # Check year-type files
    years_types_dir = os.path.join(main_dir, "years_types")
    if not os.path.exists(years_types_dir):
        print(f"ERROR: Year-type directory {years_types_dir} not found")
        return 0
    
    # Create a directory for index caches
    index_cache_dir = os.path.join(main_dir, "index_cache")
    if not os.path.exists(index_cache_dir):
        os.makedirs(index_cache_dir)
    
    # Get already processed patents to support resuming
    processed_patents = get_processed_patents(output_file)
    if processed_patents:
        print(f"Found {len(processed_patents)} already processed patent-type combinations, will skip these")
    
    # Group cited patents by year and type
    cited_by_year_type = defaultdict(list)
    
    for pnr, ptype, pyr in cited_patents:
        # Skip already processed
        if (pnr) in processed_patents:
            continue
            
        year = pyr if pyr else patent_to_year.get(pnr, None)
        if year:
            key = (year, ptype)
            cited_by_year_type[key].append((pnr, ptype, year))
        else:
            print(f"Warning: No year found for cited patent {pnr}")
    
    print(f"Organized {sum(len(v) for v in cited_by_year_type.values())} unprocessed cited patents into {len(cited_by_year_type)} year-type groups")
    
    # OPTIMIZATION: Create cache files for position indices
    needed_files = set()
    for year, ptype in cited_by_year_type.keys():
        file_path = os.path.join(years_types_dir, f"patents_{year}_{ptype}.txt")
        if not os.path.exists(file_path):
            print(f"Warning: No data file found for year {year}, type {ptype}")
            continue
        needed_files.add(file_path)
    
    print(f"Creating position index cache files for {len(needed_files)} needed patent files...")
    
    # Prepare tasks by breaking down into smaller batches
    tasks = []
    
    for (year, ptype), patents_in_group in cited_by_year_type.items():
        file_path = os.path.join(years_types_dir, f"patents_{year}_{ptype}.txt")
        if not os.path.exists(file_path):
            print(f"Warning: No data file found for year {year}, type {ptype}")
            continue
        
        # Create cache file path
        cache_file = os.path.join(index_cache_dir, f"patents_{year}_{ptype}.txt.index")
        
        # Create index cache if it doesn't exist
        if not os.path.exists(cache_file) or os.path.getsize(cache_file) == 0:
            print(f"Creating position index cache for {year}_{ptype}...")
            
            # Create the position index - using the codified IDs (base-50)
            position_index = {}
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = 0
                while True:
                    pos = f.tell()  # Record position before reading
                    line = f.readline()
                    if not line:  # End of file
                        break
                        
                    parts = line.strip().split(' ', 2)
                    if len(parts) >= 1:
                        patent_id = parts[0]  # Already codified ID (base-50)
                        position_index[patent_id] = pos
                    
                    line_count += 1
                    if line_count % 1000000 == 0:
                        print(f"  Processed {line_count} patents for index creation...")
            
            # Write the position index to a binary file for efficiency
            # Format: codified_patent_id position
            with open(cache_file, 'w', encoding='utf-8') as f:
                for patent_id, pos in position_index.items():
                    f.write(f"{patent_id} {pos}\n")
            
            print(f"Created position index cache with {len(position_index)} patents")
            # Free memory
            del position_index
        else:
            print(f"Using existing index cache for {year}_{ptype}")
        
        # Break patents_in_group into smaller batches
        for i in range(0, len(patents_in_group), batch_size):
            batch = patents_in_group[i:i+batch_size]
            # Store the cache file path with the task
            tasks.append((year, ptype, file_path, batch, top_n, None, cache_file))
    
    # Determine number of processes
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print(f"Processing {len(tasks)} batches using {num_processes} processes...")
    
    # Prepare output file
    if not os.path.exists(output_file) or len(processed_patents) == 0:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Modified header to remove patent_type
            f.write("cited_pnr control_pnr rank similarity\n")
        print(f"Created new output file: {output_file}")
    else:
        print(f"Appending to existing output file: {output_file}")
    
    # Process tasks in parallel
    total_matches = 0
    
    # Start the worker processes
    with mp.Pool(processes=num_processes, initializer=init_worker, 
                initargs=(main_dir, codified_to_original, original_to_codified, None)) as pool:
        # Submit all tasks asynchronously
        worker_results = []
        
        for i, task in enumerate(tasks):
            # Create a copy of the task with the worker_id
            worker_id = f"worker_{i % num_processes}"
            task_with_id = task[:5] + (worker_id,) + task[6:]
            result = pool.apply_async(process_cited_patent_batch_with_cache, (task_with_id,))
            worker_results.append(result)
        
        # Collect results and write directly to output file
        with open(output_file, 'a', encoding='utf-8') as f_out:
            for i, result in enumerate(worker_results):
                try:
                    # Get results from worker (only Top-N for each cited patent)
                    batch_results = result.get()
                    batch_matches = len(batch_results)
                    print(f"Batch {i+1}/{len(tasks)} completed: found {batch_matches} matches")
                    
                    # Write results directly to output file - without patent_type
                    for cited_pnr, control_pnr, rank, similarity in batch_results:
                        f_out.write(f"{cited_pnr} {control_pnr} {rank} {similarity}\n")
                    
                    # Update match count
                    total_matches += batch_matches
                    
                except Exception as e:
                    print(f"Error processing batch {i+1}: {e}")
    
    print(f"Position index cache summary:")
    print(f"- Cache files stored in: {index_cache_dir}")
    print(f"- Total cache files: {len(needed_files)}")
    
    return total_matches


def process_cited_patent_batch_with_cache(task_data):
    """
    Process a batch of cited patents against patents in a year-type file,
    using cached position index.
    """
    if len(task_data) >= 7:
        year, patent_type, file_path, cited_patents_batch, top_n, worker_id, cache_file = task_data
    else:
        year, patent_type, file_path, cited_patents_batch, top_n, worker_id = task_data
        # For backward compatibility, if no cache file is provided
        cache_file = None
    
    global codified_to_original, original_to_codified
    
    print(f"Worker {worker_id}: Processing {len(cited_patents_batch)} cited patents for {year}_{patent_type}")
    
    # Load the position index from cache
    position_index = {}
    if cache_file and os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    patent_id, pos = parts
                    position_index[patent_id] = int(pos)
    else:
        # Fallback to creating the index if cache is not available
        print(f"Worker {worker_id}: No cache file found, creating position index...")
        with open(file_path, 'r', encoding='utf-8') as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                    
                parts = line.strip().split(' ', 2)
                if len(parts) >= 1:
                    patent_id = parts[0]  # Codified ID
                    position_index[patent_id] = pos
    
    # Initialize patent cache for this worker
    patent_cache = {}
    
    # Track statistics
    total_processed = 0
    total_matches = 0
    
    # Results to return directly (only Top-N for each patent)
    all_results = []
    
    # Process each cited patent in the batch
    for cited_info in cited_patents_batch:
        cited_pnr, cited_type, cited_year = cited_info
        
        # Skip if patent type doesn't match
        if cited_type != patent_type:
            continue
        
        # Check if this cited patent is in the file (look up by original ID)
        if cited_pnr in original_to_codified:
            cited_id_codified = original_to_codified[cited_pnr]
            
            # Get cited patent tokens using position index
            cited_tokens = get_patent_tokens_with_index(cited_id_codified, file_path, position_index, patent_cache)
            
            if cited_tokens:
                # Store similarities in a min-heap for top-n selection
                top_similarities = []
                
                # Compare with all patents in the file
                for patent_id in position_index:
                    # Skip self comparison
                    if patent_id == cited_id_codified:
                        continue
                    
                    # Get comparison patent tokens
                    patent_tokens = get_patent_tokens_with_index(patent_id, file_path, position_index, patent_cache)
                    
                    if patent_tokens:
                        # Calculate similarity
                        similarity = calculate_jaccard_similarity(cited_tokens, patent_tokens)
                        
                        if similarity > 0:
                            # Convert codified ID to original
                            if patent_id in codified_to_original:
                                original_id = codified_to_original[patent_id]
                                
                                # Use min-heap to track top similarities
                                if len(top_similarities) < top_n:
                                    heapq.heappush(top_similarities, (similarity, original_id))
                                elif similarity > top_similarities[0][0]:
                                    heapq.heappop(top_similarities)
                                    heapq.heappush(top_similarities, (similarity, original_id))
                
                # Add top-N results directly to the return list
                if top_similarities:
                    # Sort by similarity (descending)
                    sorted_similarities = sorted(top_similarities, reverse=True)
                    
                    # Add each match to results
                    for rank, (similarity, control_id) in enumerate(sorted_similarities, 1):
                        all_results.append((cited_pnr, control_id, rank, similarity))
                        total_matches += 1
        
        total_processed += 1
        
        # Report progress periodically
        if total_processed % 100 == 0:
            print(f"Worker {worker_id}: Processed {total_processed}/{len(cited_patents_batch)} patents, found {total_matches} matches")
    
    # Clean up memory
    del patent_cache
    del position_index
    gc.collect()
    
    # Return only the top-N results list
    return all_results


def get_patent_tokens_with_index(patent_id, file_path, position_index, patent_cache, max_cache_size=10000):
    """
    Get patent tokens from file using position index and caching.
    Uses the provided position index instead of creating a new one.
    """
    # Check cache first
    if patent_id in patent_cache:
        return patent_cache[patent_id]
    
    # If position is known in the index, read from file
    if patent_id in position_index:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.seek(position_index[patent_id])
            line = f.readline()
            parts = line.strip().split(' ', 2)
            
            if len(parts) >= 3:
                # Extract tokens
                tokens_part = parts[2]
                token_entries = tokens_part.split(' ')
                tokens = set()
                
                for entry in token_entries:
                    if ':' in entry:
                        token, _ = entry.split(':', 1)
                        tokens.add(token)
                
                # Update cache, maintaining maximum size
                if len(patent_cache) >= max_cache_size:
                    # Remove least recently used item (first key)
                    patent_cache.pop(next(iter(patent_cache)))
                
                patent_cache[patent_id] = tokens
                return tokens
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Patent Citation Similarity Pipeline (Direct Top-N)')
    parser.add_argument('--patents', required=True, help='CSV file with patent data')
    parser.add_argument('--citations', required=True, help='CSV file with citation data')
    parser.add_argument('--dir', required=True, help='Working directory for intermediate files')
    parser.add_argument('--output', help='Output file for results (default: cited_similarity.txt)')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes to use')
    parser.add_argument('--top', type=int, default=3, help='Number of top similar patents to find (default: 3)')
    parser.add_argument('--batch', type=int, default=1000, help='Number of patents per batch (default: 1000)')
    parser.add_argument('--skip-preprocess', action='store_true', help='Skip preprocessing stages')
    args = parser.parse_args()
    
    # Set output file
    output_file = args.output or os.path.join(args.dir, "cited_similarity.txt")
    
    # Create working directory if needed
    ensure_dir_exists(args.dir)
    
    # Start timer
    start_time = time.time()
    
    # Run preprocessing stages if not skipped
    if not args.skip_preprocess:
        # Stage 1: Preprocess data
        if not run_stage1(args.dir, args.patents):
            print("Preprocessing failed. Exiting.")
            return
        
        # Stages 2-4: Codify, index, and split by year
        if not run_stages_2_to_4(args.dir):
            print("Preprocessing stages 2-4 failed. Exiting.")
            return
    
    # Create patent type index and year-type split
    patent_to_type, patent_to_year, codified_to_original, original_to_codified = create_patent_type_index(args.patents, args.dir)
    if not patent_to_type:
        print("Failed to create patent type index. Exiting.")
        return
    
    # Load cited patents
    cited_patents = load_cited_patents(args.citations)
    if not cited_patents:
        print("Failed to load cited patents. Exiting.")
        return
    
    # Find similar patents with direct Top-N results
    total_matches = find_similar_patents_direct_top_n(
        cited_patents,
        args.dir,
        patent_to_year,
        patent_to_type,
        original_to_codified,
        codified_to_original,
        output_file,
        args.top,
        args.processes,
        args.batch
    )
    
    if total_matches == 0:
        print("No similar patents found.")
        return
    
    # Report statistics
    elapsed_time = time.time() - start_time
    
    print("\nResults:")
    print(f"- Total similar patent pairs found: {total_matches}")
    print(f"- Results written to: {output_file}")
    print(f"- Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")


if __name__ == "__main__":
    main()