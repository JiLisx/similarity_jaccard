# Patent Similarity Analysis

This repository contains a set of Python scripts for analyzing patent similarity using text matching techniques.

This implementation is based on:
Arts, S., Cassiman, B., & Gomez, J. C. (2017). Text matching to measure patent similarity. Strategic Management Journal.

## Scripts Overview

### Main Scripts

- **main_script.py**: Main pipeline for patent similarity analysis by filing year. Executes all stages from preprocessing to similarity calculation.
- **main_citation_control.py**: Finds similar patents for citation control using multipra zocessing.
- **main_citation_pair.py**: Calculates similarity between patent citation pairs.

### Pipeline Stages

- **Stage01PreprocessData.py**: Preprocesses patent data from CSV format into a bag-of-words representation.
- **Stage02CodifyIdxPatents.py**: Converts patent numbers and vocabulary words into a space-efficient base-50 encoding.
- **Stage03IndexPatents.py**: Indexes patents using codified numbers and vocabulary to save disk/memory space.
- **Stage04SplitDataPerYear.py**: Splits indexed patent data by year for more efficient processing.
- **Stage05ComputeSimilarity.py**: Computes pairwise Jaccard similarity between patents with multiprocessing.

### Supporting Files

- **StopWords.py**: Base class for stopword management.
- **EnglishStopWords.py**: English stopwords implementation for filtering common words.

## Usage Examples

### 1. Main Pipeline

To run the full patent similarity pipeline:

```bash
python main_script.py --dir /path/to/workdir --start 1985 --end 2024 --processes 4
```

Options:
- `--dir`: Working directory for data files
- `--start`: Start year for similarity calculation
- `--end`: End year for similarity calculation
- `--processes`: Number of processes to use
- `--top`: Only keep top N most similar patent pairs per year
- `--stage`: Start from specific stage (1-5, 0 for all)

### 2. Citation Control

To find similar patents for citation control:

```bash
python main_citation_control.py --patents patents.csv --citations citations.csv --dir /path/to/workdir --output results.txt
```

Options:
- `--patents`: CSV file with patent data
- `--citations`: CSV file with citation data
- `--dir`: Working directory
- `--output`: Output file for results
- `--processes`: Number of processes
- `--top`: Number of top similar patents to find
- `--batch`: Number of patents per batch
- `--skip-preprocess`: Skip preprocessing stages

### 3. Citation Pair Similarity

To calculate similarity between citation pairs:

```bash
python main_citation_pair.py --patents patents.csv --citations citations.csv --dir /path/to/workdir --output similarity_results.txt
```

Options:
- `--patents`: CSV file with patent data
- `--citations`: CSV file with citation pairs
- `--output`: Output file for similarity results
- `--dir`: Working directory
- `--processes`: Number of processes to use

## Required Data Formats

### Patent Data CSV Format
The main patent data file should be a CSV with the following columns:
- `pnr`: Patent number (unique identifier)
- `year`: Publication year
- `title_en` or column at index 2: Patent title
- `abstract_en` or column at index 3: Patent abstract

Example:
```
pnr,year,title_en,abstract_en
CN101,2015,Machine learning algorithm,Advanced machine learning algorithm for pattern recognition with neural network implementation.,,
CN102,2015,Deep learning framework,Deep learning framework enabling computational efficiency for large scale data processing tasks.,,

```

### Citation Data CSV Format
Citation data should be a CSV with the following columns:
- For main_citation_control.py:
  - `cited_pnr`: Cited patent number
  - `cited_patent_type`: Type of cited patent (pnra, pnrbc or utility) (pnra refers to application patents, pnrbc refers to grant inventions, utility refers to ulitily inventions)
  - `year` (optional): Filing year of cited patent

- For main_citation_pair.py:
  - `citing_pnr`: Citing patent publication number
  - `cited_pnr`: Cited patent publication number
