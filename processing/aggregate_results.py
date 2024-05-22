import os
import argparse
import numpy as np
import pandas as pd

classes = ["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire"]
metrics = ["AUPRO@30\%", "AUPRO@10\%", "AUPRO@5\%", "AUPRO@1\%", "P-AUROC", "I-AUROC"]

def read_md_files(directory):
    aggregated_results = []
    md_files = sorted([filename for filename in os.listdir(directory) if filename.endswith(".md")])

    for filename in md_files:
        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding = "utf-8") as file:
            file_contents = file.read().split('\n')[-1].split('&') # Take only the last line (results) and split in string numbers.
            file_contents_num = [float(res) for res in file_contents] # Convert string numbers in float numbers.
            aggregated_results.append(file_contents_num)
    return np.array(aggregated_results) # Expected shape (10,6)

def produce_table(args):
    data = read_md_files(args.quantitative_folder)
    results = pd.DataFrame(data, index=classes, columns=metrics).T
    results['mean'] = results.mean(axis=1)

    print(results.to_latex(float_format = "%.3f"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Create a LaTeX table with all the results.')

    parser.add_argument('--quantitative_folder', default = None, type = str,
                        help = 'Path to the folder from which to fetch the quantitatives.')

    args = parser.parse_args()

    produce_table(args)