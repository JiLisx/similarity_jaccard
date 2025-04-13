"""
# Date: Created on Apr 12, 2025 
# Author: Ji Li (Modified with multiprocessing support)

@cite Arts, S., Cassiman, B., & Gomez, J. C. (2017). Text matching to measure patent similarity. Strategic Management Journal.

Compute pair-wise Jaccard similarity between patents in the same year with multiprocessing support.
"""
import os
import time
import multiprocessing as mp
from collections import OrderedDict, defaultdict
from functools import partial

class Stage05ComputeSimilarity:
    
    def __init__(self, num_processes=None):
        """
        Constructor
        
        Args:
            num_processes: Number of processes to use for multiprocessing.
                          If None, use available CPU count.
        """
        self.num_processes = num_processes if num_processes is not None else mp.cpu_count()
    
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
                patent = OrderedDict()
                
                for i in range(2, len(line_split)):
                    elements = line_split[i].split(':')
                    token = elements[0]
                    tf_token = int(elements[1])
                    
                    patent[token] = tf_token
                    
                    if token not in inverted_index:
                        inverted_index[token] = []
                    
                    inverted_index[token].append(id_patent)
                
                patents[id_patent] = patent
                n_patent += 1
                
                if n_patent % 10000 == 0:  # Outputs the progress of this process
                    print(f"\t{n_patent} patents read...")
    
    def process_patents(self, patent_a, values, patents, inverted_index, previous_patents):
        """
        Computes the number of keywords shared by two patents (intersection).
        It uses the inverted index to compute the intersection for all the patents related with
        a focus patent (patent A). It stores the intersections in a map.
        Since the similarities are computed pair-wise and the similarity between A and B
        is the same as between B and A, it stores a map of previous patents
        for which the similarity was already computed.
        
        Args:
            patent_a: The focus patent.
            values: The map to store all the intersections.
            patents: The patent data.
            inverted_index: The inverted index of keywords and their associated patent numbers.
            previous_patents: The map of previous patents.
        """
        for idx in patent_a:
            for idx_patent in inverted_index.get(idx, []):
                if idx_patent not in previous_patents:
                    if idx_patent not in values:
                        values[idx_patent] = 0
                    values[idx_patent] += 1
    
    def calculate_similarities_for_patent(self, args):
        """
        Calculate Jaccard similarities for a single patent against other patents.
        This function is designed to be used with multiprocessing.
        
        Args:
            args: Tuple containing (idx_patent_a, patent_a, num_kw_patent_a, patents, 
                                   inverted_index, previous_patents, lhm_patents_idx)
        
        Returns:
            List of similarity tuples (patent_a, patent_b, similarity)
        """
        idx_patent_a, patent_a, num_kw_patent_a, patents, inverted_index, previous_patents, lhm_patents_idx = args
        
        values = {}
        results = []
        
        # Process this patent against all others not in previous_patents
        for idx in patent_a:
            for idx_patent in inverted_index.get(idx, []):
                if idx_patent not in previous_patents:
                    if idx_patent not in values:
                        values[idx_patent] = 0
                    values[idx_patent] += 1
        
        # Calculate similarity for each patent pair
        for idx_patent_b, intersection in values.items():
            if idx_patent_b not in previous_patents:
                patent_b = patents[idx_patent_b]
                num_kw_patent_b = len(patent_b)
                
                union = (num_kw_patent_a + num_kw_patent_b) - intersection
                jaccard_similarity = intersection / union
                
                # Round to 6 digits
                round_sim = int(jaccard_similarity * 100000)
                jaccard_similarity = round_sim / 100000.0
                
                if jaccard_similarity > 0:  # Outputs only values greater than 0
                    results.append((lhm_patents_idx[idx_patent_a], lhm_patents_idx[idx_patent_b], jaccard_similarity))
        
        return results
    
    def jaccard_similarity(self, patents, inverted_index, f_similarity, lhm_patents_idx):
        """
        Computes the pair-wise Jaccard similarity between patents using multiprocessing.
        
        Args:
            patents: The patent data.
            inverted_index: The inverted index of keywords and their associated patent numbers.
            f_similarity: The file to store the computed similarities.
            lhm_patents_idx: The map containing the codified patent numbers.
        """
        previous_patents = OrderedDict()
        all_patent_ids = list(patents.keys())
        total_patents = len(all_patent_ids)
        
        print(f"Computing similarities for {total_patents} patents using {self.num_processes} processes...")
        
        # Prepare arguments for multiprocessing
        batch_args = []
        for idx_patent_a in all_patent_ids:
            patent_a = patents[idx_patent_a]
            num_kw_patent_a = len(patent_a)
            batch_args.append((idx_patent_a, patent_a, num_kw_patent_a, patents, 
                              inverted_index, previous_patents.copy(), lhm_patents_idx))
            # Update previous_patents for subsequent batches
            previous_patents[idx_patent_a] = 0
        
        # Use multiprocessing to compute similarities
        results = []
        processed = 0
        start_time = time.time()
        
        # Create a pool of worker processes
        with mp.Pool(processes=self.num_processes) as pool:
            # Process patents in chunks for better progress reporting
            chunk_size = max(1, min(1000, total_patents // (self.num_processes * 10)))
            
            # Use imap_unordered for better load balancing
            for batch_results in pool.imap_unordered(self.calculate_similarities_for_patent, 
                                                    batch_args, chunksize=chunk_size):
                results.extend(batch_results)
                processed += 1
                
                if processed % 1000 == 0 or processed == total_patents:
                    elapsed = time.time() - start_time
                    patents_per_sec = processed / elapsed if elapsed > 0 else 0
                    progress = (processed / total_patents) * 100
                    print(f"\t\tProcessed: {processed}/{total_patents} patents ({progress:.1f}%), "
                          f"Speed: {patents_per_sec:.1f} patents/sec, "
                          f"Found: {len(results)} similarities")
        
        print(f"Writing {len(results)} similarities to file...")
        
        # Write results to file
        with open(f_similarity, 'w', encoding='utf-8') as pw_similarity:
            for patent_a, patent_b, similarity in results:
                pw_similarity.write(f"{patent_a} {patent_b} {similarity}\n")
    
    def jaccard_similarity_sequential(self, patents, inverted_index, f_similarity, lhm_patents_idx):
        """
        Original sequential implementation of Jaccard similarity computation.
        Kept for reference and comparison.
        
        Args:
            patents: The patent data.
            inverted_index: The inverted index of keywords and their associated patent numbers.
            f_similarity: The file to store the computed similarities.
            lhm_patents_idx: The map containing the codified patent numbers.
        """
        values = OrderedDict()
        previous_patents = OrderedDict()
        n = 0
        
        # Initialize values
        for idx_patent in patents:
            values[idx_patent] = 0
        
        with open(f_similarity, 'w', encoding='utf-8') as pw_similarity:
            for idx_patent_a in patents:
                previous_patents[idx_patent_a] = 0
                patent_a = patents[idx_patent_a]
                num_kw_patent_a = len(patent_a)
                
                values.clear()
                self.process_patents(patent_a, values, patents, inverted_index, previous_patents)
                
                for idx_patent_b, intersection in values.items():
                    if idx_patent_b not in previous_patents:
                        patent_b = patents[idx_patent_b]
                        num_kw_patent_b = len(patent_b)
                        
                        union = (num_kw_patent_a + num_kw_patent_b) - intersection
                        jaccard_similarity = intersection / union
                        
                        # Round to 6 digits
                        round_sim = int(jaccard_similarity * 100000)
                        jaccard_similarity = round_sim / 100000.0
                        
                        if jaccard_similarity > 0:  # Outputs only values greater than 0
                            pw_similarity.write(f"{lhm_patents_idx[idx_patent_a]} {lhm_patents_idx[idx_patent_b]} {jaccard_similarity}\n")
                
                n += 1
                if n % 10000 == 0:  # Outputs the progress of this process
                    print(f"\t\t{n} patents processed...")

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compute patent similarity using multiprocessing")
    parser.add_argument("--dir", required=True, help="Working directory for data files")
    parser.add_argument("--start", type=int, default=2001, help="Start year for similarity calculation")
    parser.add_argument("--end", type=int, default=2003, help="End year for similarity calculation")
    parser.add_argument("--processes", type=int, default=None, help="Number of processes to use (default: all available cores)")
    parser.add_argument("--sequential", action="store_true", help="Use sequential processing instead of multiprocessing")
    
    args = parser.parse_args()
    
    main_dir = args.dir
    init_year = args.start
    end_year = args.end
    
    # Initialize with specified number of processes
    cs = Stage05ComputeSimilarity(num_processes=args.processes)
    
    f_patents_idxs = os.path.join(main_dir, "patents_idxs.txt")
    f_jaccard = os.path.join(main_dir, "jaccard")
    
    if not os.path.exists(f_jaccard):
        os.makedirs(f_jaccard)
    
    # Create dictionaries to store patent data and inverted index
    patents = OrderedDict()
    inverted_index = defaultdict(list)
    lhm_patents_idx = OrderedDict()
    
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
            
            if args.sequential:
                print("\tUsing sequential processing")
                cs.jaccard_similarity_sequential(patents, inverted_index, f_similarity, lhm_patents_idx)
            else:
                print(f"\tUsing {cs.num_processes} processes")
                cs.jaccard_similarity(patents, inverted_index, f_similarity, lhm_patents_idx)
            
            year_elapsed = time.time() - year_start_time
            print(f"\tCompleted year {year} in {year_elapsed:.2f} seconds ({year_elapsed/60:.2f} minutes)")
        else:
            print(f"\tNo data file found for year {year}")
    
    total_elapsed = time.time() - total_start_time
    print(f"Total execution time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
