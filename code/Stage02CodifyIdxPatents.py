"""
# Date: Created on Apr 8, 2025 
# Author: Ji Li

@cite Arts, S., Cassiman, B., & Gomez, J. C. (2017). Text matching to measure patent similarity. Strategic Management Journal.

Codifies the patent numbers and words in the vocabulary using a base 50 index.
This helps saving space when storing the similarity calculations
"""
import os

class Stage02CodifyIdxPatents:
    
    def __init__(self):
        self.digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                       'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
                       'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
                       'u', 'v', 'w', 'x', 'y', 'z', '!', '@', '#', '$', 
                       '%', '&', '*', '(', ')', '-', '_', '=', '+', '?']
    
    def get_digits(self):
        """
        Returns the 50 possible digits.
        
        Returns:
            The list of digits
        """
        return self.digits
    
    def convert_to_code(self, n):
        """
        Converts an integer in a base 50 code.
        
        Args:
            n: The integer to convert
            
        Returns:
            The codified integer as string.
        """
        coded = ""
        rem = 0
        if n == 0:
            coded = "0"
        else:
            while n > 0:
                rem = n % len(self.get_digits())
                coded = self.get_digits()[rem] + coded
                n = n // len(self.get_digits())
        return coded

if __name__ == "__main__":
    cip = Stage02CodifyIdxPatents()
    
    # Replace with your working directory
    main_dir = "/path/to/your/data"
    
    f_patents_num = os.path.join(main_dir, "vocabulary_raw.txt")  # Original vocabulary
    f_patents_idxs = os.path.join(main_dir, "vocabulary.txt")  # Codified vocabulary
    
    print("Codifying vocabulary...")
    with open(f_patents_num, 'r', encoding='utf-8') as br_content, \
         open(f_patents_idxs, 'w', encoding='utf-8') as pw_indexed:
        
        n = 0
        for line in br_content:
            code = cip.convert_to_code(n)
            pw_indexed.write(f"{code} {line.strip()}\n")  # Stores the codified word and the original one
            n += 1
            if n % 10000 == 0:  # Outputs the progress of this process
                print(f"\tProcessed = {n} words")
    
    f_patents_num = os.path.join(main_dir, "patents_numbers.txt")  # Original patent numbers
    f_patents_idxs = os.path.join(main_dir, "patents_idxs.txt")  # Codified patent numbers
    
    print("Codifying patent numbers...")
    with open(f_patents_num, 'r', encoding='utf-8') as br_content, \
         open(f_patents_idxs, 'w', encoding='utf-8') as pw_indexed:
        
        n = 0
        for line in br_content:
            code = cip.convert_to_code(n)
            pw_indexed.write(f"{code} {line.strip()}\n")  # Stores the codified patent number and the original one
            n += 1
            if n % 100000 == 0:  # Outputs the progress of this process
                print(f"\tProcessed = {n} patents")
