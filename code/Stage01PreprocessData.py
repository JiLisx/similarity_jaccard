"""
# Date: Created on Apr 8, 2025 
# Author: Ji Li

@cite Arts, S., Cassiman, B., & Gomez, J. C. (2017). Text matching to measure patent similarity. Strategic Management Journal.

Preprocesses patent data in a CSV format
"""
import re
import os
import csv
from collections import OrderedDict
from EnglishStopWords import EnglishStopWords

class Stage01PreprocessData:
    
    def __init__(self):
        pass

    def get_token_pattern(self):
        """
        Gets a token pattern to extract words from a text using a defined regular expression.
        The default regular expression matches alphanumeric sequences of characters and - and
        does not consider the _.
        
        Returns:
            A token pattern.
        """
        # Using a regex that matches alphanumeric sequences with hyphens but not underscores
        # Slight adaptation of the Java regex to work in Python
        return re.compile(r'\b[a-zA-Z0-9][-a-zA-Z0-9]*[a-zA-Z0-9]\b')

    def tokenize(self, text):
        """
        Splits a text into a list of tokens (words) using a defined regular expression.
        Transforms each token (word) to lower case.
        
        Args:
            text: The text to tokenize.
            
        Returns:
            A list of tokens.
        """
        tokens = []
        matcher = self.get_token_pattern().finditer(text)
        for match in matcher:
            tokens.append(match.group().lower())
        return tokens

    def create_bag_of_words(self, f_input, f_output):
        """
        Reads a file containing patent raw content and transform it into a bag-of-words file.
        The file with the raw content should have a patent per line in a CSV format.
        Takes the title and the abstract sections.
        Transform the patent content to lower case.
        By default this process removes English stopwords, words formed only by numbers
        and words of only one character.
        
        Args:
            f_input: The file containing the patents raw content.
            f_output: The file containing the patents content as a bag-of-words.
        """
        sw = EnglishStopWords()  # Stopword list
        n_docs = 0
        
        with open(f_input, 'r', encoding='utf-8') as br_content, \
             open(f_output, 'w', encoding='utf-8') as pw_output:
            
            # Skip header if exists
            next(br_content, None)
            
            csv_reader = csv.reader(br_content)
            for row in csv_reader:
                if len(row) >= 4:
                    patent_num = row[0]
                    year = row[1]
                    # Concatenate title and abstract
                    text = row[2] + " " + row[3]
                    
                    text = text.lower()
                    tokens = self.tokenize(text)
                    # Use OrderedDict to maintain lexicographical order
                    vector = OrderedDict()
                    
                    for token in tokens:
                        # Remove stopwords, words formed only by numbers and words of only one character
                        if (not sw.is_stop_word(token) and 
                            len(token) > 1 and 
                            not token.isdigit() and  # Filter pure numbers
                            not re.match(r'[0-9]+(?:-[0-9]+)+$', token)):
                            vector[token] = 1
                    
                    if vector:
                        pw_output.write(f"{patent_num} {year}")
                        # Sort tokens alphabetically
                        for token in sorted(vector.keys()):
                            pw_output.write(f" {token}")
                        pw_output.write("\n")
                    else:
                        print(n_docs)
                    
                    n_docs += 1
                    if n_docs % 100000 == 0:
                        print(f"\tProcessed = {n_docs} documents")

    def extract_vocabulary(self, f_input, f_vocabulary, threshold):
        """
        Extracts a vocabulary from a patent file in bag-of-words format.
        Prunes the vocabulary using a threshold of the minimum number of documents where a word
        should occur.
        Stores the vocabulary in a file with one word per line.
        
        Args:
            f_input: The patent file in bag-of-words format.
            f_vocabulary: The file where to store the extracted vocabulary.
            threshold: The threshold of minimum number of documents where a word should occur.
        """
        vocabulary = {}
        n_docs = 0
        
        with open(f_input, 'r', encoding='utf-8') as br_content:
            for line in br_content:
                line_split = line.strip().split(' ')
                for i in range(2, len(line_split)):
                    word = line_split[i]
                    vocabulary[word] = vocabulary.get(word, 0) + 1
                
                n_docs += 1
                if n_docs % 100000 == 0:
                    print(f"\tProcessed = {n_docs} documents")
        
        # Prune vocabulary
        self.prune_vocabulary(vocabulary, threshold)
        
        with open(f_vocabulary, 'w', encoding='utf-8') as pw_vocabulary:
            for word in vocabulary:
                pw_vocabulary.write(f"{word}\n")

    def prune_vocabulary(self, vocabulary, threshold):
        """
        Removes words from a vocabulary, given a threshold of the minimum number of documents
        where a word occurs.
        
        Args:
            vocabulary: The dictionary from where to remove words.
            threshold: The threshold of minimum number of documents where a word should occur.
        """
        words = list(vocabulary.keys())
        for word in words:
            total_frequency = vocabulary[word]
            if total_frequency < threshold:
                del vocabulary[word]

    def read_vocabulary(self, f_vocabulary, vocabulary):
        """
        Loads a vocabulary from a vocabulary file. This file must have one word per line.
        The vocabulary is formed by a word and an index, corresponding to the line number in the file.
        
        Args:
            f_vocabulary: The vocabulary file.
            vocabulary: The vocabulary dictionary to populate.
        """
        with open(f_vocabulary, 'r', encoding='utf-8') as br_vocabulary:
            n_line = 0
            for line in br_vocabulary:
                vocabulary[line.strip()] = n_line
                n_line += 1

    def clean_patents(self, f_input, f_output, f_year, f_idx, vocabulary):
        """
        Cleans a patent file in bag-of-words format by removing words that are not in the general
        vocabulary.
        Additionally creates two files, one containing the patent numbers and the year of each patent.
        The clean patent file, the patent number file and the patent year file have a correspondence one
        to one.
        
        Args:
            f_input: The patent file in bag-of-words format.
            f_output: The clean patent file.
            f_year: The patent years file.
            f_idx: The patent numbers file.
            vocabulary: The vocabulary that is used to clean the bag-of-words patent file.
        """
        n_docs = 0
        
        with open(f_input, 'r', encoding='utf-8') as br_content, \
             open(f_year, 'w', encoding='utf-8') as pw_year, \
             open(f_idx, 'w', encoding='utf-8') as pw_idx, \
             open(f_output, 'w', encoding='utf-8') as pw_output:
            
            for line in br_content:
                line_split = line.strip().split(' ')
                num_patent = line_split[0]
                year = line_split[1]
                
                vector = []
                for i in range(2, len(line_split)):
                    token = line_split[i]
                    if token in vocabulary:
                        vector.append(token)
                
                if vector:
                    pw_year.write(f"{year}\n")
                    pw_idx.write(f"{num_patent}\n")
                    pw_output.write(f"{num_patent};{len(vector)};{vector[0]}")
                    for i in range(1, len(vector)):
                        pw_output.write(f" {vector[i]}")
                    pw_output.write("\n")
                
                n_docs += 1
                if n_docs % 100000 == 0:
                    print(f"\tProcessed = {n_docs} documents")

if __name__ == "__main__":
    ppd = Stage01PreprocessData()
    
    # Replace with your working directory
    main_dir = "/path/to/your/data"
    
    f_data = os.path.join(main_dir, "patent_data_raw.csv")
    f_terms = os.path.join(main_dir, "patents_terms_raw.txt")
    f_years = os.path.join(main_dir, "patents_years.txt")
    f_patents_idxs = os.path.join(main_dir, "patents_numbers.txt")
    f_clean = os.path.join(main_dir, "patents_terms.txt")
    f_vocabulary = os.path.join(main_dir, "vocabulary_raw.txt")
    
    print("Creating bag-of-words file...")
    ppd.create_bag_of_words(f_data, f_terms)
    
    print("Extracting vocabulary...")
    ppd.extract_vocabulary(f_terms, f_vocabulary, 2)
    
    print("Cleaning the bag-of-words file...")
    vocabulary = {}
    with open(f_vocabulary, 'r', encoding='utf-8') as f:
        for line in f:
            vocabulary[line.strip()] = 1
    
    ppd.clean_patents(f_terms, f_clean, f_years, f_patents_idxs, vocabulary)
