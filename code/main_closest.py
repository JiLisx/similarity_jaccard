"""
Example usage of the patent similarity data with multiprocessing support to:
1. Find the closest matching patents
2. Create a case-control group for a set of patents
"""
import os
import argparse
import pandas as pd
import time
import multiprocessing as mp
from functools import partial
from collections import defaultdict

def process_similarity_chunk(chunk_data, min_similarity=0.05, top_n=1):
    """
    Process a chunk of similarity data to find closest matches.
    
    Args:
        chunk_data: List of lines from similarity file
        min_similarity: Minimum Jaccard similarity threshold
        top_n: Number of closest matches to find for each patent
        
    Returns:
        Dictionary mapping patent to list of closest matches
    """
    patent_matches = defaultdict(list)
    
    for line in chunk_data:
        parts = line.strip().split(' ')
        if len(parts) == 3:
            patent_a, patent_b, similarity = parts[0], parts[1], float(parts[2])
            
            # Skip self-matches (a patent matching itself)
            if patent_a == patent_b:
                continue
                
            if similarity >= min_similarity:
                # Add B as a match for A
                patent_matches[patent_a].append((patent_b, similarity))
                # Add A as a match for B
                patent_matches[patent_b].append((patent_a, similarity))
    
    # Keep only top N matches for each patent
    for patent in patent_matches:
        patent_matches[patent] = sorted(patent_matches[patent], key=lambda x: x[1], reverse=True)[:top_n]
    
    return patent_matches

def find_closest_matches(similarity_file, output_file, min_similarity=0.05, top_n=1, num_processes=None):
    """
    Find the closest matching patents for each patent in the similarity data using multiprocessing.
    
    Args:
        similarity_file: File containing patent similarity data
        output_file: File to save the results
        min_similarity: Minimum Jaccard similarity threshold
        top_n: Number of closest matches to find for each patent
        num_processes: Number of processes to use (default: all available cores)
    """
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print(f"Finding closest {top_n} matches in {similarity_file} using {num_processes} processes...")
    start_time = time.time()
    
    # Check file size to determine appropriate chunk size
    file_size = os.path.getsize(similarity_file)
    lines_to_read = min(1000000, max(100000, file_size // (500 * num_processes)))
    
    # Read all lines from the file
    with open(similarity_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    total_lines = len(all_lines)
    print(f"Total similarity entries: {total_lines}")
    
    # Split data into chunks for parallel processing
    chunks = [all_lines[i:i + lines_to_read] for i in range(0, total_lines, lines_to_read)]
    print(f"Processing data in {len(chunks)} chunks")
    
    # Process chunks in parallel
    partial_process = partial(process_similarity_chunk, min_similarity=min_similarity, top_n=top_n)
    
    patent_matches = defaultdict(list)
    with mp.Pool(processes=num_processes) as pool:
        chunk_results = list(pool.map(partial_process, chunks))
    
    # Merge chunk results
    for chunk_result in chunk_results:
        for patent, matches in chunk_result.items():
            patent_matches[patent].extend(matches)
    
    # Sort and keep top N for each patent after merging all results
    for patent in patent_matches:
        patent_matches[patent] = sorted(patent_matches[patent], key=lambda x: x[1], reverse=True)[:top_n]
    
    # Write results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Patent,MatchingPatent,JaccardIndex\n")
        for patent, matches in patent_matches.items():
            for match_patent, similarity in matches:
                f.write(f"{patent},{match_patent},{similarity:.6f}\n")
    
    elapsed_time = time.time() - start_time
    print(f"Processed {total_lines} similarity entries in {elapsed_time:.2f} seconds")
    print(f"Found matches for {len(patent_matches)} patents")
    print(f"Saved results to {output_file}")

def process_target_chunk(chunk_data, similarity_file, target_patents, min_similarity=0.05):
    """
    Process a chunk of similarity data to find matches for target patents.
    
    Args:
        chunk_data: List of lines from similarity file
        similarity_file: Name of the similarity file (for reporting)
        target_patents: Set of target patents
        min_similarity: Minimum similarity threshold
        
    Returns:
        Dictionary of target patents to matching control patents
    """
    case_control_matches = defaultdict(list)
    
    for line in chunk_data:
        parts = line.strip().split(' ')
        if len(parts) == 3:
            patent_a, patent_b, similarity = parts[0], parts[1], float(parts[2])
            
            # Skip self-matches
            if patent_a == patent_b:
                continue
                
            if similarity >= min_similarity:
                # If A is a target patent, add B as potential control
                if patent_a in target_patents and patent_b not in target_patents:
                    case_control_matches[patent_a].append((patent_b, similarity))
                
                # If B is a target patent, add A as potential control
                if patent_b in target_patents and patent_a not in target_patents:
                    case_control_matches[patent_b].append((patent_a, similarity))
    
    return case_control_matches

def process_similarity_file(similarity_file, target_patents, min_similarity=0.05, num_processes=None):
    """
    Process a single similarity file to find matches for target patents.
    
    Args:
        similarity_file: File containing patent similarity data
        target_patents: Set of target patents
        min_similarity: Minimum similarity threshold
        num_processes: Number of processes to use
        
    Returns:
        Dictionary of target patents to matching control patents
    """
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Check file size to determine appropriate chunk size
    file_size = os.path.getsize(similarity_file)
    lines_to_read = min(1000000, max(100000, file_size // (500 * num_processes)))
    
    # Read all lines from the file
    with open(similarity_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    total_lines = len(all_lines)
    
    # Split data into chunks for parallel processing
    chunks = [all_lines[i:i + lines_to_read] for i in range(0, total_lines, lines_to_read)]
    
    # Process chunks in parallel
    partial_process = partial(process_target_chunk, 
                             similarity_file=similarity_file,
                             target_patents=target_patents,
                             min_similarity=min_similarity)
    
    case_control_matches = defaultdict(list)
    with mp.Pool(processes=num_processes) as pool:
        chunk_results = list(pool.map(partial_process, chunks))
    
    # Merge chunk results
    for chunk_result in chunk_results:
        for patent, matches in chunk_result.items():
            case_control_matches[patent].extend(matches)
    
    return case_control_matches

def create_case_control_group(target_patents_file, similarity_files, output_file, min_similarity=0.05, max_matches=1, num_processes=None):
    """
    Create a case-control group for a set of target patents using multiprocessing.
    
    Args:
        target_patents_file: File containing the list of target patents
        similarity_files: List of files containing patent similarity data
        output_file: File to save the case-control group
        min_similarity: Minimum Jaccard similarity threshold
        max_matches: Maximum number of control patents per target patent
        num_processes: Number of processes to use (default: all available cores)
    """
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print(f"Creating case-control group for patents in {target_patents_file} using {num_processes} processes...")
    start_time = time.time()
    
    # Read target patents
    target_patents = set()
    with open(target_patents_file, 'r', encoding='utf-8') as f:
        for line in f:
            target_patents.add(line.strip())
    
    print(f"Loaded {len(target_patents)} target patents")
    
    # Dictionary to store patent -> [(control patent, similarity), ...]
    case_control_matches = defaultdict(list)
    
    # Process each similarity file
    for similarity_file in similarity_files:
        print(f"Processing {similarity_file}...")
        file_matches = process_similarity_file(similarity_file, 
                                             target_patents, 
                                             min_similarity, 
                                             num_processes)
        
        # Merge results from this file
        for patent, matches in file_matches.items():
            case_control_matches[patent].extend(matches)
    
    # Sort matches by similarity and keep top matches
    for patent, matches in case_control_matches.items():
        case_control_matches[patent] = sorted(matches, key=lambda x: x[1], reverse=True)[:max_matches]
    
    # Write results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("TargetPatent,ControlPatent,JaccardIndex\n")
        for patent, matches in case_control_matches.items():
            for match_patent, similarity in matches:
                f.write(f"{patent},{match_patent},{similarity:.6f}\n")
    
    # Report statistics
    matched_patents = sum(1 for matches in case_control_matches.values() if matches)
    elapsed_time = time.time() - start_time
    
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"Found control patents for {matched_patents} out of {len(target_patents)} target patents")
    print(f"Saved case-control pairs to {output_file}")

def analyze_similarity_data(jaccard_dir, output_dir, year_range, num_processes=None):
    """
    Analyze the similarity data and generate reports using multiprocessing.
    
    Args:
        jaccard_dir: Directory containing Jaccard similarity files
        output_dir: Directory to save analysis results
        year_range: Range of years to analyze (tuple of start_year, end_year)
        num_processes: Number of processes to use (default: all available cores)
    """
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    start_year, end_year = year_range
    print(f"Analyzing similarity data for years {start_year}-{end_year} using {num_processes} processes...")
    
    # Calculate statistics per year
    stats = []
    
    def analyze_year(year):
        """Analyze a single year's similarity data"""
        similarity_file = os.path.join(jaccard_dir, f"jaccard_{year}.txt")
        
        if not os.path.exists(similarity_file):
            return None
        
        print(f"Analyzing similarity data for {year}...")
        
        # Count patents and similarities
        patents = set()
        similarity_count = 0
        avg_similarity = 0
        min_sim = float('inf')
        max_sim = 0
        
        with open(similarity_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) == 3:
                    patent_a, patent_b, similarity = parts[0], parts[1], float(parts[2])
                    
                    # Skip self-matches for stats
                    if patent_a == patent_b:
                        continue
                        
                    patents.add(patent_a)
                    patents.add(patent_b)
                    similarity_count += 1
                    avg_similarity += similarity
                    min_sim = min(min_sim, similarity)
                    max_sim = max(max_sim, similarity)
        
        if similarity_count > 0:
            avg_similarity /= similarity_count
            return {
                'Year': year,
                'Patents': len(patents),
                'Similarities': similarity_count,
                'AvgSimilarity': avg_similarity,
                'MinSimilarity': min_sim,
                'MaxSimilarity': max_sim
            }
        return None
    
    # Process years in parallel
    years = list(range(start_year, end_year + 1))
    with mp.Pool(processes=num_processes) as pool:
        year_results = pool.map(analyze_year, years)
    
    # Filter out None results and combine
    stats = [result for result in year_results if result is not None]
    
    # Create summary report
    if stats:
        df = pd.DataFrame(stats)
        summary_file = os.path.join(output_dir, "similarity_stats.csv")
        df.to_csv(summary_file, index=False)
        print(f"Saved similarity statistics to {summary_file}")
        
        # Print summary
        print("\nSummary of Similarity Data:")
        print(f"Years analyzed: {start_year} to {end_year}")
        print(f"Total patents: {df['Patents'].sum()}")
        print(f"Total similarities: {df['Similarities'].sum()}")
        print(f"Average similarity: {df['AvgSimilarity'].mean():.6f}")
        
        # Create closest matches file for each year (in parallel)
        def process_year_closest(year):
            similarity_file = os.path.join(jaccard_dir, f"jaccard_{year}.txt")
            if os.path.exists(similarity_file):
                closest_match_file = os.path.join(output_dir, f"closest_match_{year}.csv")
                find_closest_matches(similarity_file, closest_match_file, num_processes=1)  # Use 1 process within each parallel job
                return year
            return None
        
        print("\nGenerating closest matches for each year...")
        with mp.Pool(processes=min(len(years), num_processes)) as pool:
            processed_years = pool.map(process_year_closest, years)
        
        processed_years = [y for y in processed_years if y is not None]
        print(f"Generated closest matches files for {len(processed_years)} years")

def main():
    parser = argparse.ArgumentParser(description='Patent Similarity Analysis Tools with Multiprocessing')
    parser.add_argument('--jaccard_dir', type=str, required=True, help='Directory containing Jaccard similarity files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files')
    parser.add_argument('--start_year', type=int, default=2001, help='Start year for analysis')
    parser.add_argument('--end_year', type=int, default=2003, help='End year for analysis')
    parser.add_argument('--target_patents', type=str, help='File containing target patents for case-control analysis')
    parser.add_argument('--action', type=str, choices=['analyze', 'closest', 'case_control'], 
                        default='analyze', help='Action to perform')
    parser.add_argument('--processes', type=int, default=None, 
                        help='Number of processes to use (default: all available cores)')
    parser.add_argument('--min_similarity', type=float, default=0.05,
                        help='Minimum similarity threshold (default: 0.05)')
    parser.add_argument('--top_n', type=int, default=1,
                        help='Number of top matches to find (default: 1)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    start_time = time.time()
    
    if args.action == 'analyze':
        analyze_similarity_data(args.jaccard_dir, args.output_dir, 
                               (args.start_year, args.end_year), 
                               args.processes)
    
    elif args.action == 'closest':
        # Find closest matches for each year
        for year in range(args.start_year, args.end_year + 1):
            similarity_file = os.path.join(args.jaccard_dir, f"jaccard_{year}.txt")
            if os.path.exists(similarity_file):
                output_file = os.path.join(args.output_dir, f"closest_match_{year}.csv")
                find_closest_matches(similarity_file, output_file, 
                                    args.min_similarity, args.top_n, args.processes)
    
    elif args.action == 'case_control':
        if not args.target_patents:
            print("Error: --target_patents file must be specified for case_control action")
            return
        
        # Collect all similarity files
        similarity_files = []
        for year in range(args.start_year, args.end_year + 1):
            similarity_file = os.path.join(args.jaccard_dir, f"jaccard_{year}.txt")
            if os.path.exists(similarity_file):
                similarity_files.append(similarity_file)
        
        output_file = os.path.join(args.output_dir, "case_control_pairs.csv")
        create_case_control_group(args.target_patents, similarity_files, output_file, 
                                 args.min_similarity, args.top_n, args.processes)
    
    total_elapsed = time.time() - start_time
    print(f"\nTotal execution time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")

if __name__ == "__main__":
    main()
