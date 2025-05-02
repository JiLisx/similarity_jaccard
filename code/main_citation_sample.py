"""
Memory-Optimized Patent Control Group Finder

This script finds the top 5 most similar patents from the same year AND SAME TYPE for each cited patent
in a citation file, creating a control group for further analysis. It uses streaming approach
to handle very large patent databases with minimal memory footprint.

Usage:
    python patent_control_finder.py --patents patents_database.csv --citations citation_pnr_CN_sample.csv 
                                   --output control_groups.csv --processes 8 --cache-size 10000

Author: Ji Li
Date: May 2, 2025
"""

import os
import csv
import time
import argparse
import re
import multiprocessing as mp
from collections import defaultdict, OrderedDict, deque
import heapq
from functools import partial
import threading
import sys

# Import required components from the existing codebase
from EnglishStopWords import EnglishStopWords


# Create a file lock for safe concurrent writing
file_lock = threading.Lock()


class PatentControlFinder:
    
    def __init__(self, num_processes=None, top_n=5, cache_size=10000):
        """
        Initialize the patent control finder with memory optimization
        
        Args:
            num_processes: Number of processes to use (None = use all available cores)
            top_n: Number of most similar patents to find for each cited patent
            cache_size: Maximum number of patents to keep in memory cache
        """
        self.num_processes = num_processes if num_processes is not None else mp.cpu_count()
        self.top_n = top_n
        self.sw = EnglishStopWords()  # Stop words for preprocessing
        self.max_cache_size = cache_size
        self.patent_cache = {}  # LRU cache for patent data
        self.cache_usage_queue = deque()  # For tracking LRU order
    
    def get_token_pattern(self):
        """
        Define regex pattern to extract valid tokens from text
        """
        return re.compile(r'\b[a-zA-Z0-9][-a-zA-Z0-9]*[a-zA-Z0-9]\b')
    
    def tokenize(self, text):
        """
        Split text into tokens and convert to lowercase
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        if not text or not isinstance(text, str):
            return []
            
        tokens = []
        matcher = self.get_token_pattern().finditer(text)
        for match in matcher:
            tokens.append(match.group().lower())
        return tokens
    
    def preprocess_patent_text(self, title, abstract):
        """
        Create a clean set of tokens from patent title and abstract
        
        Args:
            title: Patent title
            abstract: Patent abstract
            
        Returns:
            Set of clean tokens
        """
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
        
        return clean_tokens  # 添加了返回值
    
    def _update_cache(self, patent_id, patent_data):
        """
        Update the patent cache using LRU strategy
        
        Args:
            patent_id: Patent ID to add/update in cache
            patent_data: Patent data to cache
        """
        # If patent is already in cache, remove it from the usage queue
        if patent_id in self.patent_cache:
            self.cache_usage_queue.remove(patent_id)
        # If cache is full, remove least recently used patent
        elif len(self.patent_cache) >= self.max_cache_size:
            oldest_id = self.cache_usage_queue.popleft()
            del self.patent_cache[oldest_id]
        
        # Add/update patent in cache and mark as most recently used
        self.patent_cache[patent_id] = patent_data
        self.cache_usage_queue.append(patent_id)
    
    def create_patent_index(self, patents_file):
        """
        Create an index of patents organized by year AND patent type for efficient lookup
        
        Args:
            patents_file: CSV file with patent data
            
        Returns:
            Dictionary mapping (year, type) pairs to patent lists, positions, etc.
        """
        print(f"Creating patent index for {patents_file}...")
        start_time = time.time()
        
        # Organize patents by BOTH year AND type
        patents_by_year_and_type = defaultdict(list)
        patent_positions = {}
        type_counts = defaultdict(int)
        
        with open(patents_file, 'r', encoding='utf-8') as f:
            # Get header and its position
            header_pos = f.tell()
            header_row = f.readline().strip().split(',')
            
            # Find column indices
            pnr_idx = header_row.index('pnr') if 'pnr' in header_row else 0
            year_idx = header_row.index('year') if 'year' in header_row else 1
            
            # Check for required patent_type column
            type_idx = -1
            for possible_name in ['patent_type', 'type', 'pnr_type']:
                if possible_name in header_row:
                    type_idx = header_row.index(possible_name)
                    break
            
            if type_idx == -1:
                print("ERROR: Required 'patent_type' column not found in patent database file.")
                print("Column names must include one of: 'patent_type', 'type', or 'pnr_type'")
                sys.exit(1)
            
            # Index each patent's position by year and type
            patent_count = 0
            errors = 0
            while True:
                position = f.tell()
                line = f.readline()
                if not line:
                    break
                
                csv_reader = csv.reader([line.strip()])
                parts = next(csv_reader)
                if len(parts) > max(pnr_idx, year_idx, type_idx):
                    patent_id = parts[pnr_idx].strip()
                    year = parts[year_idx].strip()
                    patent_type = parts[type_idx].strip()
                    
                    # Store patent position and index by year AND type
                    if patent_id and year and patent_type:
                        patent_positions[patent_id] = position
                        year_type_key = (year, patent_type.lower())  # 直接使用原始类型，只转为小写
                        patents_by_year_and_type[year_type_key].append(patent_id)
                        patent_count += 1
                        type_counts[patent_type.lower()] += 1
                    else:
                        errors += 1
                        if errors < 10:  # Only show first few errors
                            print(f"Warning: Missing essential info for patent in line: {line}")
                        elif errors == 10:
                            print("Suppressing further warnings...")
                
                # Show progress
                if patent_count % 100000 == 0:
                    elapsed = time.time() - start_time
                    print(f"Indexed {patent_count} patents in {elapsed:.2f} seconds")
        
        if errors > 0:
            print(f"WARNING: {errors} patents had invalid or missing information and were skipped")
        
        elapsed = time.time() - start_time
        print(f"Created index for {patent_count} patents in {elapsed:.2f} seconds")
        
        # Report on patent types
        print("Patent types found:")
        for patent_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {patent_type}: {count} patents")
        
        # Report on years with most patents
        year_stats = defaultdict(int)
        for (year, _), patents in patents_by_year_and_type.items():
            year_stats[year] += len(patents)
            
        top_years = sorted([(year, count) for year, count in year_stats.items()], 
                          key=lambda x: x[1], reverse=True)[:5]
        print("Years with most patents:")
        for year, count in top_years:
            print(f"  {year}: {count} patents")
        
        column_indices = {
            'pnr': pnr_idx,
            'year': year_idx,
            'type': type_idx,
            'title': header_row.index('title_en') if 'title_en' in header_row else 2,
            'abstract': header_row.index('abstract_en') if 'abstract_en' in header_row else 3
        }
        
        return patents_by_year_and_type, patent_positions, header_pos, column_indices
    
    def get_patent_data(self, patent_id, patents_file, patent_positions, header_pos, column_indices):
        """
        Retrieve patent data for a specific patent ID, using cache for efficiency
        
        Args:
            patent_id: Patent ID to retrieve
            patents_file: CSV file with patent data
            patent_positions: Dictionary mapping patent IDs to file positions
            header_pos: Position of the header row in the file
            column_indices: Dictionary mapping column names to their indices
            
        Returns:
            Dictionary with patent data or None if not found
        """
        # Check cache first
        if patent_id in self.patent_cache:
            return self.patent_cache[patent_id]
        
        # If not in cache, retrieve from file
        if patent_id in patent_positions:
            with open(patents_file, 'r', encoding='utf-8') as f:
                # Jump to the patent's position
                f.seek(patent_positions[patent_id])
                patent_line = f.readline().strip()
                
                # Parse the patent data
                csv_reader = csv.reader([patent_line])
                parts = next(csv_reader)
                
                # Get indices for columns
                pnr_idx = column_indices['pnr']
                year_idx = column_indices['year']
                type_idx = column_indices['type']
                title_idx = column_indices['title']
                abstract_idx = column_indices['abstract']
                
                if len(parts) > max(pnr_idx, year_idx, type_idx, title_idx):
                    # Extract fields
                    patent_id = parts[pnr_idx].strip()
                    year = parts[year_idx].strip()
                    patent_type = parts[type_idx].strip().lower()  # 直接使用原始类型，只转为小写
                    title = parts[title_idx].strip() if title_idx < len(parts) else ""
                    
                    # Handle the case where the abstract might contain commas
                    if abstract_idx < len(parts):
                        # This is a simplification - real CSV parsing is more complex
                        abstract = ','.join(parts[abstract_idx:]).strip()
                        # Remove quotes if present
                        if abstract.startswith('"') and abstract.endswith('"'):
                            abstract = abstract[1:-1]
                    else:
                        abstract = ""
                    
                    # Preprocess text
                    tokens = self.preprocess_patent_text(title, abstract)
                    
                    # Skip patents with no valid tokens
                    if not tokens:
                        return None
                    
                    # Create patent data
                    patent_data = {
                        'year': year,
                        'type': patent_type,
                        'tokens': tokens
                    }
                    
                    # Update cache with this patent
                    self._update_cache(patent_id, patent_data)
                    
                    return patent_data
        
        # Patent not found
        return None
    
    def load_cited_patents(self, citations_file):
        """
        Load cited patents from the citations file along with their types
        
        Args:
            citations_file: CSV file containing citation data
            
        Returns:
            List of cited patent tuples (patent_id, type_info)
        """
        print(f"Loading cited patents from {citations_file}...")
        cited_patents_with_type = []
        
        with open(citations_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            
            # Check which columns we have
            has_type_column = 'cited_patent_type' in header
            
            if not has_type_column:
                print("ERROR: Required 'cited_patent_type' column not found in citations file")
                print("Citations CSV must include a 'cited_patent_type' column with values like:")
                print("  - 'pnrA' for invention patents")
                print("  - 'pnrBC' or 'utility' for utility model patents")
                sys.exit(1)
            
            # Reset to beginning and read citations
            f.seek(0)
            reader = csv.DictReader(f)
            
            for row in reader:
                # Get cited patent ID and type
                cited_pnr = row.get('cited_pnr', row.get('cited_patent', ''))
                cited_type = row.get('cited_patent_type', '')
                
                if cited_pnr and cited_type:
                    cited_tuple = (cited_pnr, cited_type.lower())  # 直接使用原始类型，只转为小写
                    if cited_tuple not in cited_patents_with_type:
                        cited_patents_with_type.append(cited_tuple)
                else:
                    print(f"Warning: Skipping cited patent without ID or type: {row}")
        
        print(f"Loaded {len(cited_patents_with_type)} unique cited patents with type information")
        return cited_patents_with_type
    
    def calculate_jaccard_similarity(self, set_a, set_b):
        """
        Calculate Jaccard similarity between two sets of tokens
        
        Args:
            set_a: First set of tokens
            set_b: Second set of tokens
            
        Returns:
            Jaccard similarity (0-1)
        """
        if not set_a or not set_b:
            return 0.0
            
        intersection = len(set_a.intersection(set_b))
        if intersection == 0:
            return 0.0
            
        union = len(set_a.union(set_b))
        return intersection / union
    
    def process_patent_batch(self, batch_data):
        """
        Process a batch of cited patents to find similar patents and write results directly to file
        
        Args:
            batch_data: Tuple containing:
                - list of cited patents with types to process
                - patents_by_year_and_type: dictionary mapping (year, type) to patent lists
                - patents_file: file containing patent data
                - patent_positions: dictionary mapping patent IDs to file positions
                - header_pos: position of header in file
                - column_indices: dictionary mapping column names to indices
                - output_file: file to write results to
                - top_n: number of top matches to find
            
        Returns:
            Dictionary with statistics about processed patents
        """
        cited_patents_with_type, patents_by_year_and_type, patents_file, patent_positions, header_pos, column_indices, output_file, top_n = batch_data
        
        # Statistics to return
        stats = {
            'processed': 0,
            'found': 0,
            'matches': 0
        }
        
        # Create local caches for this process
        local_cache = {}  # Local cache for this process
        
        # Create a list to store results before writing
        batch_results = []
        
        for cited_pnr, cited_type in cited_patents_with_type:
            # Get cited patent data
            cited_data = self.get_patent_data(cited_pnr, patents_file, patent_positions, header_pos, column_indices)
            
            # Skip if cited patent not in our database
            if not cited_data:
                stats['processed'] += 1
                continue
                
            # Get year and tokens for cited patent (type already determined)
            cited_year = cited_data['year']
            cited_tokens = cited_data['tokens']
            
            # Get candidate patents from the same year AND same type
            year_type_key = (cited_year, cited_type)
            same_year_type_patents = patents_by_year_and_type.get(year_type_key, [])
            
            # Find similarities with other patents from the same year and type
            similarities = []
            
            for candidate_pnr in same_year_type_patents:
                # Skip self-comparison
                if candidate_pnr == cited_pnr:
                    continue
                
                # Check local cache first
                if candidate_pnr in local_cache:
                    candidate_tokens = local_cache[candidate_pnr]['tokens']
                else:
                    # Get candidate patent data
                    candidate_data = self.get_patent_data(candidate_pnr, patents_file, patent_positions, 
                                                         header_pos, column_indices)
                    
                    # Skip if candidate patent data not available
                    if not candidate_data:
                        continue
                        
                    candidate_tokens = candidate_data['tokens']
                    
                    # Update local cache (with size limit)
                    if len(local_cache) < 1000:  # Limit local cache size
                        local_cache[candidate_pnr] = candidate_data
                
                # Calculate similarity
                similarity = self.calculate_jaccard_similarity(cited_tokens, candidate_tokens)
                
                # Only keep if similarity is above zero
                if similarity > 0:
                    # Use negative similarity for max-heap behavior using min-heap
                    heapq.heappush(similarities, (-similarity, candidate_pnr))
                    
                    # If we've exceeded top_n, remove the lowest similarity
                    if len(similarities) > top_n:
                        heapq.heappop(similarities)
            
            # Convert to list of (patent_id, similarity) tuples, sorted by similarity (highest first)
            top_matches = [(pnr, -sim) for sim, pnr in sorted(similarities)]
            
            if top_matches:  # Only add to results if matches were found
                stats['found'] += 1
                stats['matches'] += len(top_matches)
                
                # Add to batch results
                for rank, (control_pnr, similarity) in enumerate(top_matches, 1):
                    batch_results.append([cited_pnr, control_pnr, rank, f"{similarity:.6f}", cited_type])
            
            stats['processed'] += 1
        
        # Write results to file with lock to prevent conflicts between processes
        if batch_results:
            with file_lock:
                with open(output_file, 'a', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(batch_results)
        
        return stats
    
    def find_control_patents_streaming(self, cited_patents_with_type, patents_by_year_and_type, patents_file, 
                                      patent_positions, header_pos, column_indices, output_file):
        """
        Find the top N most similar patents for each cited patent using a streaming approach
        that minimizes memory usage and writes results directly to file.
        
        Args:
            cited_patents_with_type: List of cited patent tuples (patent_id, type_info)
            patents_by_year_and_type: Dictionary mapping (year, type) to patent lists
            patents_file: File containing patent data
            patent_positions: Dictionary mapping patent IDs to file positions
            header_pos: Position of header in file
            column_indices: Dictionary mapping column names to indices
            output_file: File to write results to
        """
        print(f"Finding top {self.top_n} similar patents for {len(cited_patents_with_type)} cited patents...")
        start_time = time.time()
        
        # Create output file with header
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['cited_pnr', 'control_pnr', 'rank', 'similarity', 'patent_type'])
        
        # Process in batches for better multiprocessing performance
        batch_size = max(50, min(500, len(cited_patents_with_type) // (self.num_processes * 2)))
        if batch_size == 0 and len(cited_patents_with_type) > 0:
            batch_size = len(cited_patents_with_type)
            
        batches = []
        
        for i in range(0, len(cited_patents_with_type), batch_size):
            batch = cited_patents_with_type[i:i + batch_size]
            batches.append((
                batch, 
                patents_by_year_and_type, 
                patents_file, 
                patent_positions, 
                header_pos, 
                column_indices, 
                output_file,
                self.top_n
            ))
        
        # Process batches in parallel
        total_stats = {
            'processed': 0,
            'found': 0,
            'matches': 0
        }
        
        print(f"Processing in {len(batches)} batches of ~{batch_size} patents each using {self.num_processes} processes...")
        
        with mp.Pool(processes=self.num_processes) as pool:
            batch_count = 0
            for batch_stats in pool.imap_unordered(self.process_patent_batch, batches):
                batch_count += 1
                
                # Update total statistics
                for key in total_stats:
                    total_stats[key] += batch_stats[key]
                
                # Output progress
                if batch_count % max(1, len(batches) // 10) == 0 or batch_count == len(batches):
                    elapsed = time.time() - start_time
                    patents_per_sec = total_stats['processed'] / elapsed if elapsed > 0 else 0
                    
                    print(f"Processed {batch_count}/{len(batches)} batches ({batch_count*100/len(batches):.1f}%), "
                          f"found controls for {total_stats['found']}/{total_stats['processed']} patents, "
                          f"speed: {patents_per_sec:.1f} patents/sec")
        
        elapsed = time.time() - start_time
        print(f"Found control patents for {total_stats['found']}/{total_stats['processed']} "
              f"cited patents in {elapsed:.2f} seconds")
        
        return total_stats
    
    def analyze_results(self, output_file):
        """
        Analyze the results file to provide statistics
        
        Args:
            output_file: Output CSV file path
        """
        print(f"Analyzing results in {output_file}...")
        
        try:
            cited_patents = set()
            control_patents = set()
            matches_count = 0
            type_stats = defaultdict(int)
            
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header
                
                # Check if we have patent type in results
                has_type_column = len(header) >= 5
                
                for row in reader:
                    if len(row) >= 4:
                        cited_pnr, control_pnr, rank, similarity = row[:4]
                        cited_patents.add(cited_pnr)
                        control_patents.add(control_pnr)
                        matches_count += 1
                        
                        # Track patent type statistics if available
                        if has_type_column and len(row) >= 5:
                            patent_type = row[4]
                            type_stats[patent_type] += 1
            
            print(f"Results statistics:")
            print(f"- Total cited patents with matches: {len(cited_patents)}")
            print(f"- Total unique control patents: {len(control_patents)}")
            print(f"- Total matches: {matches_count}")
            print(f"- Average matches per cited patent: {matches_count / len(cited_patents) if cited_patents else 0:.2f}")
            
            if type_stats:
                print(f"- Patent type distribution:")
                for patent_type, count in sorted(type_stats.items(), key=lambda x: x[1], reverse=True):
                    print(f"  * {patent_type}: {count} matches")
            
        except Exception as e:
            print(f"Error analyzing results: {e}")


def main():
    parser = argparse.ArgumentParser(description='Find control patents for cited patents based on text similarity')
    parser.add_argument('--patents', required=True, help='CSV file with patent data (pnr,year,patent_type,title_en,abstract_en)')
    parser.add_argument('--citations', required=True, help='CSV file with citation data (cited_pnr,cited_patent_type)')
    parser.add_argument('--output', required=True, help='Output file for control patent pairs')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes to use (default: all available cores)')
    parser.add_argument('--top', type=int, default=5, help='Number of top similar patents to find (default: 5)')
    parser.add_argument('--cache-size', type=int, default=10000, help='Maximum patents to keep in memory cache (default: 10000)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize the patent control finder
    finder = PatentControlFinder(num_processes=args.processes, top_n=args.top, cache_size=args.cache_size)
    
    # Create patent index instead of loading all patents
    patents_by_year_and_type, patent_positions, header_pos, column_indices = finder.create_patent_index(args.patents)
    
    # Load cited patents with type information
    cited_patents_with_type = finder.load_cited_patents(args.citations)
    
    # Find control patents using streaming approach with real-time writing
    finder.find_control_patents_streaming(
        cited_patents_with_type, 
        patents_by_year_and_type, 
        args.patents, 
        patent_positions, 
        header_pos, 
        column_indices,
        args.output
    )
    
    # Analyze the results
    finder.analyze_results(args.output)
    
    print("Patent control group finding completed!")


if __name__ == "__main__":
    main()