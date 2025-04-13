"""
# Date: Created on Apr 8, 2025 
# Author: Ji Li

@cite Arts, S., Cassiman, B., & Gomez, J. C. (2017). Text matching to measure patent similarity. Strategic Management Journal.

Splits the indexed patent data per year
"""
import os

class Stage04SplitDataPerYear:
    
    def __init__(self):
        """Constructor"""
        pass
    
    def split_data_per_year(self, f_content, al_years, f_main_dir):
        """
        Splits the indexed patent data per year. It creates new directory (years) inside
        the current working directory.
        
        Args:
            f_content: The file containing the indexed patent data.
            al_years: A list containing the year of each patent.
            f_main_dir: The working directory to store the split data.
        """
        years = {}  # Dictionary to store file handlers for each year
        n_patent = 0
        
        years_dir = os.path.join(f_main_dir, "years")
        print(years_dir)
        if not os.path.exists(years_dir):  # Create directory if it does not exist
            os.makedirs(years_dir)
        
        with open(f_content, 'r', encoding='utf-8') as br_content:
            for line in br_content:
                year_patent = al_years[n_patent]
                year_pat = int(year_patent)
                
                if 1900 <= year_pat <= 2025:  # Check the maximum year of the data
                    if year_patent not in years:
                        f_year = os.path.join(years_dir, f"patents_indexed_{year_patent}.txt")
                        years[year_patent] = open(f_year, 'w', encoding='utf-8')
                    
                    years[year_patent].write(line)
                else:
                    print(year_pat)
                
                n_patent += 1
                if n_patent % 100000 == 0:  # Outputs the progress of this process
                    print(f"\tProcessed {n_patent} documents...")
        
        # Close all year files
        for pw_year in years.values():
            pw_year.close()
        
        print(f"Total patents = {n_patent}")
    
    def read_years(self, f_years, years):
        """
        Read a file containing the year of each patent. The years are stored one per line, and
        there is a correspondence one to one with the indexed patent data.
        
        Args:
            f_years: The file containing the patent years.
            years: A list to store the year of each patent.
        """
        with open(f_years, 'r', encoding='utf-8') as br_content:
            for line in br_content:
                years.append(line.strip())

if __name__ == "__main__":
    sdpy = Stage04SplitDataPerYear()
    
    # Replace with your working directory
    main_dir = "/path/to/your/data"
    
    f_indexed = os.path.join(main_dir, "patents_indexed.txt")  # Indexed patent data
    f_years = os.path.join(main_dir, "patents_years.txt")  # Patent years file
    
    al_years = []
    
    print("Loading patent years...")
    sdpy.read_years(f_years, al_years)
    
    print("Splitting patent data per year...")
    sdpy.split_data_per_year(f_indexed, al_years, main_dir)
