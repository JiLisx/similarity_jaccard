"""
# Date: Created on Apr 8, 2025 
# Author: Ji Li

@cite Arts, S., Cassiman, B., & Gomez, J. C. (2017). Text matching to measure patent similarity. Strategic Management Journal.

Indexes the patent data using the codified patent numbers and words from the vocabulary.
This helps to save space in disk and memory when computing the similarities.
"""
import os
from collections import OrderedDict

class Stage03IndexPatents:
    
    def __init__(self):
        """Constructor"""
        pass
    
    def index_patents(self, f_clean, f_indexed, vocabulary, patents_idxs):
        """
        Indexes the patent data using the codified patent numbers and words from the general
        vocabulary.
        
        Args:
            f_clean: The file containing the clean patent data.
            f_indexed: The file where to store the indexed patent data.
            vocabulary: The codified vocabulary for the data.
            patents_idxs: The codified patent numbers.
        """
        n_docs = 0
        
        with open(f_clean, 'r', encoding='utf-8') as br_content, \
             open(f_indexed, 'w', encoding='utf-8') as pw_indexed:
            
            for line in br_content:
                line_split = line.strip().split(';')
                num_patent = line_split[0]
                num_terms = line_split[1]
                
                idx_patent = patents_idxs.get(num_patent)
                pw_indexed.write(f"{idx_patent} {num_terms}")
                
                terms = line_split[2].split(' ')
                for token in terms:
                    idx_token = vocabulary.get(token)
                    pw_indexed.write(f" {idx_token}:1")
                
                pw_indexed.write("\n")
                
                n_docs += 1
                if n_docs % 100000 == 0:  # Outputs the progress of this process
                    print(f"\tProcessed = {n_docs} documents")
    
    def read_indexes(self, f_content, lhm_idx):
        """
        Loads a file composed of a codified index and a original number (or word).
        Stores the codified index and the original number (or word) in a map.
        
        Args:
            f_content: The file containing codified indexes and original numbers (or words).
            lhm_idx: The map to store the data from the file.
        """
        with open(f_content, 'r', encoding='utf-8') as br_content:
            for line in br_content:
                line_break = line.strip().split(' ', 1)
                if len(line_break) == 2:
                    lhm_idx[line_break[1]] = line_break[0]

if __name__ == "__main__":
    ip = Stage03IndexPatents()
    
    # Replace with your working directory
    main_dir = "/path/to/your/data"
    
    f_clean = os.path.join(main_dir, "patents_terms.txt")  # Clean patent data
    f_patents_idxs = os.path.join(main_dir, "patents_idxs.txt")  # Codified patent numbers
    f_vocabulary = os.path.join(main_dir, "vocabulary.txt")  # Codified vocabulary
    f_indexed = os.path.join(main_dir, "patents_indexed.txt")  # Indexed patent data
    
    vocabulary = OrderedDict()
    patents_idxs = OrderedDict()
    
    print("Loading codified vocabulary...")
    ip.read_indexes(f_vocabulary, vocabulary)
    
    print("Loading codified patent numbers...")
    ip.read_indexes(f_patents_idxs, patents_idxs)
    
    print("Indexing patent data...")
    ip.index_patents(f_clean, f_indexed, vocabulary, patents_idxs)
