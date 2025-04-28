"""
# Date: Created on Apr 12, 2025 
# Author: Ji Li

@cite Arts, S., Cassiman, B., & Gomez, J. C. (2017). Text matching to measure patent similarity. Strategic Management Journal.

Compute pair-wise Jaccard similarity between patents in the same year with optimized multiprocessing support.
Now includes functionality to keep only the TOP-N most similar patent pairs to reduce storage requirements.
Compatible with Python 3.6+
"""
import os
import time
import heapq
import multiprocessing as mp
from functools import partial
from collections import defaultdict

class Stage05ComputeSimilarity:
    
    def __init__(self, num_processes=None, top_n=None):
        """
        Constructor
        
        Args:
            num_processes: Number of processes to use for multiprocessing.
                          If None, use available CPU count.
            top_n: If specified, only store the top N most similar patent pairs.
                  If None, store all similarities.
        """
        self.num_processes = num_processes if num_processes is not None else mp.cpu_count()
        self.top_n = top_n
    
    def read_indexes(self, f_input, lhm):
        """
        Reads the codified patent numbers and the original ones and stores them in a map.
        
        Args:
            f_input: The file containing the codified indexes.
            lhm: The map to store the patent numbers.
        """
        with open(f_input, 'r', encoding='utf-8') as br_input:
            for line in br_input:
                line_break = line.strip().split(' ', 1)
                if len(line_break) == 2:
                    lhm[line_break[0]] = line_break[1]
    
    def read_patents_sequentially(self, f_content, patents, inverted_index):
        """
        Loads the patent data from a file, one at a time, and stores them in a map. The key for
        the patent in the map is the codified patent number.
        
        Additionally it stores an inverted index in a map, for each keyword it stores the list of
        codified patent numbers that are associated with it.
        
        Args:
            f_content: The patent data file.
            patents: The map to store the patent data.
            inverted_index: The inverted index of keywords and their associated patent numbers.
        """
        n_patent = 0
        
        with open(f_content, 'r', encoding='utf-8') as br_content:
            for line in br_content:
                line_split = line.strip().split(' ')
                id_patent = line_split[0]
                patent = {}
                
                for i in range(2, len(line_split)):
                    elements = line_split[i].split(':')
                    token = elements[0]
                    tf_token = int(elements[1])
                    
                    patent[token] = tf_token
                    
                    inverted_index.setdefault(token, []).append(id_patent)
                
                patents[id_patent] = patent
                n_patent += 1
                
                if n_patent % 10000 == 0:  # Outputs the progress of this process
                    print(f"\t{n_patent} patents read...")
    
    def process_patent_batch(self, batch_data):
        """
        Process a batch of patents to find similarities
        
        Args:
            batch_data: Tuple containing (patent_ids, batch_start, batch_end, patents, inverted_index, top_n)
            
        Returns:
            If top_n is specified, returns the top_n most similar patent pairs.
            Otherwise, returns all similarity tuples (patent_a, patent_b, similarity)
        """
        patent_ids, batch_start, batch_end, patents, token_to_patents, top_n = batch_data
        
        results = []
        
        # For each patent in this batch
        for i in range(batch_start, batch_end):
            patent_id_a = patent_ids[i]
            patent_a = patents[patent_id_a]
            num_tokens_a = len(patent_a)
            
            # Find candidates that share at least one token
            candidates = set()
            for token in patent_a:
                if token in token_to_patents:
                    for pid in token_to_patents[token]:
                        # Only process patents with higher indices to avoid duplicates
                        if patent_ids.index(pid) > i:
                            candidates.add(pid)
            
            # Calculate similarity for each candidate
            for patent_id_b in candidates:
                patent_b = patents[patent_id_b]
                num_tokens_b = len(patent_b)
                
                # Calculate intersection (tokens in both patent_a and patent_b)
                intersection = 0
                for token in patent_a:
                    if token in patent_b:
                        intersection += 1
                
                if intersection > 0:
                    # Calculate Jaccard similarity
                    union = num_tokens_a + num_tokens_b - intersection
                    jaccard_similarity = intersection / union
                    
                    # Round to 6 digits
                    jaccard_similarity = round(jaccard_similarity, 6)
                    
                    if jaccard_similarity > 0:
                        # Use negative similarity for max-heap behavior with a min-heap
                        if top_n is not None:
                            heapq.heappush(results, (-jaccard_similarity, patent_id_a, patent_id_b))
                            # If we've exceeded top_n, remove the lowest similarity
                            if len(results) > top_n:
                                heapq.heappop(results)
                        else:
                            results.append((patent_id_a, patent_id_b, jaccard_similarity))
        
        # If using top_n, convert the results back to the expected format
        if top_n is not None:
            # Convert (-similarity, id_a, id_b) to (id_a, id_b, similarity)
            temp_results = [
                (pid_a, pid_b, -sim) for sim, pid_a, pid_b in results
            ]
            # Sort by similarity in descending order
            temp_results.sort(key=lambda x: x[2], reverse=True)
            return temp_results
        
        return results
    
    def jaccard_similarity(self, patents, inverted_index, f_similarity, lhm_patents_idx):
        """
        Computes the pair-wise Jaccard similarity between patents using optimized multiprocessing.
        If self.top_n is specified, only the top N most similar patent pairs are saved.
        
        Args:
            patents: The patent data.
            inverted_index: The inverted index of keywords and their associated patent numbers.
            f_similarity: The file to store the computed similarities.
            lhm_patents_idx: The map containing the codified patent numbers.
        """
        total_patents = len(patents)
        if self.top_n:
            print(f"Computing similarities for {total_patents} patents using {self.num_processes} processes...")
            print(f"Only storing the TOP {self.top_n} most similar patent pairs.")
        else:
            print(f"Computing similarities for {total_patents} patents using {self.num_processes} processes...")
        
        start_time = time.time()
        
        # Create a list of patent IDs to ensure consistent ordering
        patent_ids = list(patents.keys())
        
        # Optimize inverted index structure for lookups
        token_to_patents = {}
        for token, patent_list in inverted_index.items():
            token_to_patents[token] = list(set(patent_list))  # Ensure uniqueness
        
        # Divide work into batches
        batch_size = max(1, total_patents // (self.num_processes * 2))
        batches = []
        
        for start_idx in range(0, total_patents, batch_size):
            end_idx = min(start_idx + batch_size, total_patents)
            batches.append((patent_ids, start_idx, end_idx, patents, token_to_patents, self.top_n))
        
        all_similarities = []
        
        # Process batches in parallel
        with mp.Pool(processes=self.num_processes) as pool:
            batch_count = 0
            for batch_results in pool.imap_unordered(self.process_patent_batch, batches):
                all_similarities.extend(batch_results)
                batch_count += 1
                
                # Output progress
                if batch_count % max(1, len(batches) // 10) == 0:
                    print(f"\t\tProcessed {batch_count}/{len(batches)} batches, "
                          f"found {len(all_similarities)} similarities so far...")
        
        # If we're keeping the top N, we need to sort all similarities and take the top N
        if self.top_n:
            print(f"\t\tSorting {len(all_similarities)} similarities to find the top {self.top_n}...")
            all_similarities.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity in descending order
            all_similarities = all_similarities[:self.top_n]  # Keep only the top N
        
        # Write the final similarities to file
        with open(f_similarity, 'w', encoding='utf-8') as file_handle:
            for entry in all_similarities:
                if len(entry) == 3:  # Standard format (patent_id_a, patent_id_b, similarity)
                    patent_id_a, patent_id_b, similarity = entry
                else:  # For compatibility with older format if needed
                    patent_id_a, patent_id_b = entry[:2]
                    similarity = entry[2] if len(entry) > 2 else 0
                    
                original_pid_a = lhm_patents_idx.get(patent_id_a, patent_id_a)
                original_pid_b = lhm_patents_idx.get(patent_id_b, patent_id_b)
                file_handle.write(f"{original_pid_a} {original_pid_b} {similarity}\n")
        
        elapsed_time = time.time() - start_time
        print(f"Completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"Total similarities saved: {len(all_similarities)}")
    
    def jaccard_similarity_sequential(self, patents, inverted_index, f_similarity, lhm_patents_idx):
        """
        Sequential implementation of Jaccard similarity computation (optimized).
        If self.top_n is specified, only the top N most similar patent pairs are saved.
        
        Args:
            patents: The patent data.
            inverted_index: The inverted index of keywords and their associated patent numbers.
            f_similarity: The file to store the computed similarities.
            lhm_patents_idx: The map containing the codified patent numbers.
        """
        processed = 0
        total_patents = len(patents)
        start_time = time.time()
        
        # Create a list of patent IDs to ensure consistent ordering
        patent_ids = list(patents.keys())
        
        # If keeping only top N, we'll collect all similarities and sort at the end
        if self.top_n:
            print(f"Only storing the TOP {self.top_n} most similar patent pairs.")
            similarities = []
        else:
            similarities = None
            
        # Open the output file
        if not self.top_n:
            output_file = open(f_similarity, 'w', encoding='utf-8')
        
        try:
            for i, patent_id_a in enumerate(patent_ids):
                patent_a = patents[patent_id_a]
                num_tokens_a = len(patent_a)
                
                # Find all patents that share at least one token with patent_a
                candidates = set()
                for token in patent_a:
                    for patent_id in inverted_index.get(token, []):
                        # Only process patents with higher indices to avoid duplicates
                        if patent_id != patent_id_a and patent_ids.index(patent_id) > i:
                            candidates.add(patent_id)
                
                # Calculate similarity for each candidate
                for patent_id_b in candidates:
                    patent_b = patents[patent_id_b]
                    num_tokens_b = len(patent_b)
                    
                    # Calculate intersection
                    intersection = sum(1 for token in patent_a if token in patent_b)
                    
                    if intersection > 0:
                        # Calculate Jaccard similarity
                        union = num_tokens_a + num_tokens_b - intersection
                        jaccard_similarity = intersection / union
                        
                        # Round to 6 digits
                        jaccard_similarity = round(jaccard_similarity, 6)
                        
                        if jaccard_similarity > 0:  # Only process values greater than 0
                            if self.top_n:
                                # If using top_n, collect the similarities for later sorting
                                similarities.append((patent_id_a, patent_id_b, jaccard_similarity))
                                # Optional: Keep the list size manageable if it gets very large
                                if len(similarities) > self.top_n * 100:
                                    similarities.sort(key=lambda x: x[2], reverse=True)
                                    similarities = similarities[:self.top_n]
                            else:
                                # Otherwise, write to file immediately
                                original_pid_a = lhm_patents_idx.get(patent_id_a, patent_id_a)
                                original_pid_b = lhm_patents_idx.get(patent_id_b, patent_id_b)
                                output_file.write(f"{original_pid_a} {original_pid_b} {jaccard_similarity}\n")
                
                processed += 1
                if processed % 1000 == 0 or processed == total_patents:
                    elapsed = time.time() - start_time
                    patents_per_sec = processed / elapsed if elapsed > 0 else 0
                    progress = (processed / total_patents) * 100
                    if self.top_n:
                        print(f"\t\tProcessed: {processed}/{total_patents} patents ({progress:.1f}%), "
                              f"Speed: {patents_per_sec:.1f} patents/sec, Collected: {len(similarities)} similarities")
                    else:
                        print(f"\t\tProcessed: {processed}/{total_patents} patents ({progress:.1f}%), "
                              f"Speed: {patents_per_sec:.1f} patents/sec")
        
        finally:
            # Close the output file if we've been writing to it
            if not self.top_n and 'output_file' in locals():
                output_file.close()
        
        # If we're keeping only the top N, sort all similarities and save only the top N
        if self.top_n:
            print(f"\t\tSorting {len(similarities)} similarities to find the top {self.top_n}...")
            similarities.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity in descending order
            similarities = similarities[:self.top_n]  # Keep only the top N
            
            # Write the top N similarities to file
            with open(f_similarity, 'w', encoding='utf-8') as output_file:
                for patent_id_a, patent_id_b, similarity in similarities:
                    original_pid_a = lhm_patents_idx.get(patent_id_a, patent_id_a)
                    original_pid_b = lhm_patents_idx.get(patent_id_b, patent_id_b)
                    output_file.write(f"{original_pid_a} {original_pid_b} {similarity}\n")
            
            print(f"Saved the top {len(similarities)} most similar patent pairs.")

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compute patent similarity using optimized multiprocessing")
    parser.add_argument("--dir", required=True, help="Working directory for data files")
    parser.add_argument("--start", type=int, default=2001, help="Start year for similarity calculation")
    parser.add_argument("--end", type=int, default=2003, help="End year for similarity calculation")
    parser.add_argument("--processes", type=int, default=None, help="Number of processes to use (default: all available cores)")
    parser.add_argument("--top", type=int, default=None, help="Only keep the top N most similar patent pairs per year")
    
    args = parser.parse_args()
    
    main_dir = args.dir
    init_year = args.start
    end_year = args.end
    
    # Initialize with specified number of processes and top_n
    cs = Stage05ComputeSimilarity(num_processes=args.processes, top_n=args.top)
    
    f_patents_idxs = os.path.join(main_dir, "patents_idxs.txt")
    f_jaccard = os.path.join(main_dir, "jaccard")
    
    if not os.path.exists(f_jaccard):
        os.makedirs(f_jaccard)
    
    # Create dictionaries to store patent data and inverted index
    patents = {}
    inverted_index = {}
    lhm_patents_idx = {}
    
    print("Reading the codified patent numbers...")
    cs.read_indexes(f_patents_idxs, lhm_patents_idx)
    
    total_start_time = time.time()
    
    for year in range(init_year, end_year + 1):
        patents.clear()
        inverted_index.clear()
        
        print(f"Computing similarities for year = {year}")
        f_year_data = os.path.join(main_dir, f"years/patents_indexed_{year}.txt")
        
        if os.path.exists(f_year_data):
            year_start_time = time.time()
            
            print(f"\tReading data for year = {year}")
            cs.read_patents_sequentially(f_year_data, patents, inverted_index)
            
            f_similarity = os.path.join(f_jaccard, f"jaccard_{year}.txt")
            print("\tDoing the calculations...")
            print(f"\tUsing {cs.num_processes} processes")
            cs.jaccard_similarity(patents, inverted_index, f_similarity, lhm_patents_idx)
            
            year_elapsed = time.time() - year_start_time
            print(f"\tCompleted year {year} in {year_elapsed:.2f} seconds ({year_elapsed/60:.2f} minutes)")
        else:
            print(f"\tNo data file found for year {year}")
    
    total_elapsed = time.time() - total_start_time
    print(f"Total execution time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")