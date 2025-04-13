"""
# Date: Created on Apr 12, 2025 
# Author: Ji Li (Modified with enhanced multiprocessing support)

Patent Similarity Analysis 
@cite Arts, S., Cassiman, B., & Gomez, J. C. (2017). Text matching to measure patent similarity. Strategic Management Journal.

This script executes all stages of the patent similarity analysis pipeline in sequence,
with optimized multiprocessing support for the similarity computation stage.
"""

import os
import time
import argparse
from Stage01PreprocessData import Stage01PreprocessData
from Stage02CodifyIdxPatents import Stage02CodifyIdxPatents
from Stage03IndexPatents import Stage03IndexPatents
from Stage04SplitDataPerYear import Stage04SplitDataPerYear
from Stage05ComputeSimilarity import Stage05ComputeSimilarity

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def run_stage1(main_dir):
    """Preprocess patent data"""
    print("\n" + "="*80)
    print("STAGE 1: Preprocessing Patent Data")
    print("="*80)
    
    # Check if the Stage 1 output file already exists
    f_terms = os.path.join(main_dir, "patents_terms.txt")
    f_years = os.path.join(main_dir, "patents_years.txt")
    f_patents_idxs = os.path.join(main_dir, "patents_numbers.txt")
    f_vocabulary = os.path.join(main_dir, "vocabulary_raw.txt")
    
    if all(os.path.exists(f) and os.path.getsize(f) > 0 for f in [f_terms, f_years, f_patents_idxs, f_vocabulary]):
        print("Stage 1 output file already exists, skipping...")
        return True
    
    ppd = Stage01PreprocessData()
    
    f_data = os.path.join(main_dir, "patent_data_raw.csv")
    f_terms_raw = os.path.join(main_dir, "patents_terms_raw.txt")
    
    # Check if input file exists
    if not os.path.exists(f_data):
        print(f"ERROR: Input file {f_data} not found. Please place your CSV file in the working directory.")
        return False
    
    print("Creating bag-of-words file...")
    ppd.create_bag_of_words(f_data, f_terms_raw)
    
    print("Extracting vocabulary...")
    ppd.extract_vocabulary(f_terms_raw, f_vocabulary, 2)
    
    print("Cleaning the bag-of-words file...")
    vocabulary = {}
    ppd.read_vocabulary(f_vocabulary, vocabulary)
    ppd.clean_patents(f_terms_raw, f_terms, f_years, f_patents_idxs, vocabulary)
    
    return True

def run_stage2(main_dir):
    """Codify indices for patents and vocabulary"""
    print("\n" + "="*80)
    print("STAGE 2: Codifying Patent and Vocabulary Indices")
    print("="*80)
    
    # Check if the Stage 2 output file already exists
    f_vocabulary = os.path.join(main_dir, "vocabulary.txt")
    f_patents_idxs = os.path.join(main_dir, "patents_idxs.txt")
    
    if all(os.path.exists(f) and os.path.getsize(f) > 0 for f in [f_vocabulary, f_patents_idxs]):
        print("Stage 2 output file already exists, skipping...")
        return True
    
    cip = Stage02CodifyIdxPatents()
    
    f_patents_num = os.path.join(main_dir, "vocabulary_raw.txt")
    
    if not os.path.exists(f_patents_num):
        print(f"ERROR: Input file {f_patents_num} not found. Stage 1 may not have completed successfully.")
        return False
    
    print("Codifying vocabulary...")
    with open(f_patents_num, 'r', encoding='utf-8') as br_content, \
         open(f_vocabulary, 'w', encoding='utf-8') as pw_indexed:
        
        n = 0
        for line in br_content:
            code = cip.convert_to_code(n)
            pw_indexed.write(f"{code} {line.strip()}\n")
            n += 1
            if n % 10000 == 0:
                print(f"\tProcessed = {n} words")
    
    f_patents_num = os.path.join(main_dir, "patents_numbers.txt")
    
    if not os.path.exists(f_patents_num):
        print(f"ERROR: Input file {f_patents_num} not found. Stage 1 may not have completed successfully.")
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
    
    return True

def run_stage3(main_dir):
    """Index patents using codified numbers and vocabulary"""
    print("\n" + "="*80)
    print("STAGE 3: Indexing Patents")
    print("="*80)
    
    # Check if the Stage 3 output file already exists
    f_indexed = os.path.join(main_dir, "patents_indexed.txt")
    
    if os.path.exists(f_indexed) and os.path.getsize(f_indexed) > 0:
        print("Stage 3 exists, skipping ...")
        return True
    
    ip = Stage03IndexPatents()
    
    f_clean = os.path.join(main_dir, "patents_terms.txt")
    f_patents_idxs = os.path.join(main_dir, "patents_idxs.txt")
    f_vocabulary = os.path.join(main_dir, "vocabulary.txt")
    
    required_files = [f_clean, f_patents_idxs, f_vocabulary]
    for file in required_files:
        if not os.path.exists(file):
            print(f"ERROR: Input file {file} not found. Previous stages may not have completed successfully.")
            return False
    
    vocabulary = {}
    patents_idxs = {}
    
    print("Loading codified vocabulary...")
    ip.read_indexes(f_vocabulary, vocabulary)
    
    print("Loading codified patent numbers...")
    ip.read_indexes(f_patents_idxs, patents_idxs)
    
    print("Indexing patent data...")
    ip.index_patents(f_clean, f_indexed, vocabulary, patents_idxs)
    
    return True

def run_stage4(main_dir):
    """Split indexed patent data by year"""
    print("\n" + "="*80)
    print("STAGE 4: Splitting Patent Data by Year")
    print("="*80)
    
    # Check if the Stage 4 output file already exists
    years_dir = os.path.join(main_dir, "years")
    
    if os.path.exists(years_dir) and os.path.isdir(years_dir) and len(os.listdir(years_dir)) > 0:
        print("Stage 4 output file already exists, skipping...")
        return True
    
    sdpy = Stage04SplitDataPerYear()
    
    f_indexed = os.path.join(main_dir, "patents_indexed.txt")
    f_years = os.path.join(main_dir, "patents_years.txt")
    
    required_files = [f_indexed, f_years]
    for file in required_files:
        if not os.path.exists(file):
            print(f"ERROR: Input file {file} not found. Previous stages may not have completed successfully.")
            return False
    
    al_years = []
    
    print("Loading patent years...")
    sdpy.read_years(f_years, al_years)
    
    print("Splitting patent data per year...")
    sdpy.split_data_per_year(f_indexed, al_years, main_dir)
    
    return True

def run_stage5(main_dir, start_year, end_year, num_processes, sequential=False):
    """Compute Jaccard similarity for patents within each year using optimized multiprocessing"""
    print("\n" + "="*80)
    print(f"STAGE 5: Computing Jaccard Similarity (Years {start_year}-{end_year})")
    if sequential:
        print("Mode: Sequential processing")
    else:
        print(f"Mode: Multiprocessing with {num_processes if num_processes else 'all available'} processes")
    print("="*80)
    
    cs = Stage05ComputeSimilarity(num_processes=num_processes)
    
    f_patents_idxs = os.path.join(main_dir, "patents_idxs.txt")
    f_jaccard = os.path.join(main_dir, "jaccard")
    years_dir = os.path.join(main_dir, "years")
    
    if not os.path.exists(f_patents_idxs):
        print(f"ERROR: Input file {f_patents_idxs} not found. Previous stages may not have completed successfully.")
        return False
    
    if not os.path.exists(years_dir):
        print(f"ERROR: Years directory {years_dir} not found. Stage 4 may not have completed successfully.")
        return False
    
    ensure_dir_exists(f_jaccard)
    
    patents = {}
    inverted_index = {}
    lhm_patents_idx = {}
    
    print("Reading the codified patent numbers...")
    cs.read_indexes(f_patents_idxs, lhm_patents_idx)
    
    success = False
    total_start_time = time.time()
    
    for year in range(start_year, end_year + 1):
        patents.clear()
        inverted_index.clear()
        
        print(f"Computing similarities for year = {year}")
        f_year_data = os.path.join(main_dir, f"years/patents_indexed_{year}.txt")
        f_similarity = os.path.join(f_jaccard, f"jaccard_{year}.txt")
        
        # Check that the similarity file for the year already exists and is not empty
        if os.path.exists(f_similarity) and os.path.getsize(f_similarity) > 0:
            print(f"Similarity calculation for \t year {year} has been completed, skipping...")
            success = True
            continue
        
        if os.path.exists(f_year_data):
            year_start_time = time.time()
            success = True
            
            print(f"\tReading data for year = {year}")
            cs.read_patents_sequentially(f_year_data, patents, inverted_index)
            
            print("\tDoing the calculations...")
            
            if sequential:
                cs.jaccard_similarity_sequential(patents, inverted_index, f_similarity, lhm_patents_idx)
            else:
                cs.jaccard_similarity(patents, inverted_index, f_similarity, lhm_patents_idx)
            
            year_elapsed = time.time() - year_start_time
            print(f"\tCompleted year {year} in {year_elapsed:.2f} seconds ({year_elapsed/60:.2f} minutes)")
        else:
            print(f"\tNo data file found for year {year}, skipping...")
    
    total_elapsed = time.time() - total_start_time
    print(f"Stage 5 total execution time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    
    return success

def main():
    parser = argparse.ArgumentParser(description='Patent Similarity Analysis Pipeline with Optimized Multiprocessing')
    parser.add_argument('--dir', type=str, required=True, help='Working directory for data files')
    parser.add_argument('--start', type=int, default=2001, help='Start year for similarity calculation (default: 2001)')
    parser.add_argument('--end', type=int, default=2003, help='End year for similarity calculation (default: 2003)')
    parser.add_argument('--stage', type=int, default=0, help='Start from specific stage (1-5, 0 for all stages)')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes to use for multiprocessing (default: all available cores)')
    parser.add_argument('--sequential', action='store_true', help='Use sequential processing instead of multiprocessing')
    args = parser.parse_args()
    
    main_dir = args.dir
    start_year = args.start
    end_year = args.end
    start_stage = args.stage
    num_processes = args.processes
    sequential = args.sequential
    
    # Create main directory if it doesn't exist
    ensure_dir_exists(main_dir)
    
    start_time = time.time()
    
    # Run all stages or from a specific stage
    if start_stage <= 1:
        if run_stage1(main_dir):
            print("Stage 1 completed successfully!")
        else:
            print("Stage 1 failed. Exiting.")
            return
    
    if start_stage <= 2:
        if run_stage2(main_dir):
            print("Stage 2 completed successfully!")
        else:
            print("Stage 2 failed. Exiting.")
            return
    
    if start_stage <= 3:
        if run_stage3(main_dir):
            print("Stage 3 completed successfully!")
        else:
            print("Stage 3 failed. Exiting.")
            return
    
    if start_stage <= 4:
        if run_stage4(main_dir):
            print("Stage 4 completed successfully!")
        else:
            print("Stage 4 failed. Exiting.")
            return
    
    if start_stage <= 5:
        if run_stage5(main_dir, start_year, end_year, num_processes, sequential):
            print("Stage 5 completed successfully!")
        else:
            print("Stage 5 encountered issues but completed.")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print("\nPatent similarity analysis completed!")

if __name__ == "__main__":
    main()