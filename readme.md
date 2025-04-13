# Patent Similarity Analysis with Multiprocessing

This is an enhanced version of the patent similarity analysis system with multiprocessing support for significantly improved performance on multi-core systems.

## Overview

The system processes patent data through several stages:
1. **Preprocessing Data**: Transforms raw patent data into a bag-of-words format
2. **Codifying Indices**: Creates space-efficient base-50 encodings for patents and vocabulary 
3. **Indexing Patents**: Indexes patents using codified numbers and vocabulary
4. **Splitting Data by Year**: Organizes patent data by publication year
5. **Computing Similarity**: Calculates Jaccard similarity between patents (now with multiprocessing)

## Key Improvements

The multiprocessing enhancement provides:
- Parallel processing of patent similarity calculations
- Configurable number of worker processes
- Progress reporting with speed metrics
- Significant performance improvements on multi-core systems
- Option to use either parallel or sequential processing
- **New**: Smart stage skipping - automatically skips stages with existing outputs
- **New**: Force recomputation option to override stage skipping

## Requirements

- Python 3.6 or higher
- Input data in CSV format with patent number, year, title, and abstract

## Usage

### Running the Complete Pipeline

```bash
python main_script.py --dir /path/to/your/data --start 2001 --end 2003 --processes 8
```

### Arguments

- `--dir`: Working directory for data files (required)
- `--start`: Start year for similarity calculation (default: 2001)
- `--end`: End year for similarity calculation (default: 2003)
- `--stage`: Start from specific stage (1-5, 0 for all stages)
- `--processes`: Number of processes to use (default: all available cores)
- `--sequential`: Use sequential processing instead of multiprocessing


### Running Only the Similarity Calculation Stage

```bash
python Stage05ComputeSimilarity.py --dir /path/to/your/data --start 2001 --end 2003 --processes 8
```

### Running from a Specific Stage

```bash
python main_script.py --dir /path/to/your/data --stage 3 --processes 6
```

### Force Recomputation of All Stages

```bash
python main_script.py --dir /path/to/your/data --force
```

## Intelligent Stage Management

The pipeline now features intelligent file existence checks:
- Each stage automatically checks for the existence of its required output files
- If all outputs already exist, the stage is skipped with a notification
- Use the `--force` flag to override this behavior and recompute all stages

# Patent Similarity Analysis: `main_closest.py`

This tool is a high-performance utility for analyzing patent similarity data using multiprocessing capabilities. It enables researchers to find the closest matching patents, create case-control groups, and analyze similarity patterns across large patent datasets.

## Features

- **Multiprocessing Support**: Utilizes all available CPU cores for significant performance improvements
- **Multiple Analysis Modes**: Offers three distinct analysis operations
- **Memory-Efficient Processing**: Handles large datasets by processing in chunks
- **Configurable Parameters**: Customizable similarity thresholds and match counts
- **Performance Metrics**: Provides detailed timing and processing statistics

## Requirements

- Python 3.6+
- pandas
- Similarity files generated by the patent similarity pipeline (Stage05ComputeSimilarity)

## Installation

No special installation required beyond Python and the pandas library:

```bash
pip install pandas
```

## Usage

```bash
python main_closest.py --jaccard_dir <similarity_files_dir> --output_dir <results_dir> [OPTIONS]
```

### Required Arguments

- `--jaccard_dir`: Directory containing the Jaccard similarity files
- `--output_dir`: Directory where results will be saved

### Optional Arguments

- `--action`: Analysis operation to perform (default: "analyze")
  - `analyze`: Generate statistical analysis of similarity data
  - `closest`: Find closest matching patents
  - `case_control`: Create case-control groups for target patents
- `--start_year`: Starting year for analysis (default: 2001)
- `--end_year`: Ending year for analysis (default: 2003)
- `--processes`: Number of parallel processes to use (default: all available CPU cores)
- `--min_similarity`: Minimum similarity threshold (default: 0.05)
- `--top_n`: Number of closest matches to find per patent (default: 1)
- `--target_patents`: File containing target patents (required for case_control action)

## Operation Modes

### 1. Analyze Mode

Performs statistical analysis of patent similarity data and generates summary reports.

```bash
python main_closest.py --action analyze --jaccard_dir ./similarity --output_dir ./results
```

**Output**: 
- `similarity_stats.csv`: Summary statistics for each year
- Console output with aggregate statistics

### 2. Closest Mode

Finds the most similar patents for each patent in the dataset.

```bash
python main_closest.py --action closest --jaccard_dir ./similarity --output_dir ./results --top_n 5
```

**Output**: 
- `closest_match_{year}.csv`: Files containing each patent and its top N matches
- Format: "Patent,MatchingPatent,JaccardIndex"

### 3. Case-Control Mode

Creates matched control groups for a specified set of target patents.

```bash
python main_closest.py --action case_control --jaccard_dir ./similarity --output_dir ./results --target_patents ./targets.txt
```

**Output**:
- `case_control_pairs.csv`: Mapping of target patents to their control matches
- Format: "TargetPatent,ControlPatent,JaccardIndex"

## Input File Formats

### Similarity Files

The tool expects Jaccard similarity files in the format:
```
patent1 patent2 similarity_score
```

### Target Patents File

For case-control analysis, provide a file with one patent number per line:
```
US9123456
US8765432
...
```

## Advanced Examples

### Finding Top 10 Most Similar Patents with Higher Threshold

```bash
python main_closest.py --action closest --jaccard_dir ./data --output_dir ./results --top_n 10 --min_similarity 0.15 --processes 16
```

### Creating Multiple Controls per Target Patent

```bash
python main_closest.py --action case_control --jaccard_dir ./data --output_dir ./results --target_patents ./biotech_patents.txt --top_n 3 --processes 8
```

### Analyzing a Specific Time Period with Custom Process Count

```bash
python main_closest.py --action analyze --jaccard_dir ./data --output_dir ./results --start_year 2010 --end_year 2015 --processes 4
```

## Performance Considerations

- For very large datasets, increase the chunk size by modifying the `lines_to_read` variable
- The optimal number of processes depends on your CPU and available memory
- For systems with limited memory, using fewer processes may be more efficient

## Citation

If you use this tool in your research, please cite:

```
Arts, S., Cassiman, B., & Gomez, J. C. (2017). Text matching to measure patent similarity. 
Strategic Management Journal.
```

## License

This script is provided for research purposes only.

## Example Command Line

```bash
python3 ./code/main_script.py --dir ./data/ --start 1985 --end 2025

python3 ./code/main_closest.py --action closest --jaccard_dir ./data/jaccard/ --output_dir ./results --processes 2 --start_year 2001 --end_year 2020 --top_n 10 --min_similarity 0.00
```