import sys
import math
from utils import is_numerical, read_csv


def describe_data(headers, data):
    features = []
    # for each column
    for i, column in enumerate(zip(*data)):
        if is_numerical(column):
            # count the number of values
            numeric_values = [float(value) for value in column if value.replace('.', '', 1).isdigit()]
            count = len(numeric_values)
            if count == 0:
                continue
            # the average of the numbers
            mean = sum(numeric_values) / count
            # std = math.sqrt(sum([(x - mean)**2 for x in numeric_values]) / count)
            std = (sum([(x - mean) ** 2 for x in numeric_values]) / count) ** 0.5
            min_val = min(numeric_values)
            max_val = max(numeric_values)
            # order the numbers from smallest to largest, and pick the index in 25 %, 50 % and 75 %
            p_25 = sorted(numeric_values)[int(count * 0.25)]
            p_50 = sorted(numeric_values)[int(count * 0.50)]
            p_75 = sorted(numeric_values)[int(count * 0.75)]

            features.append({headers[i]: {
                "Count": count, "Mean": mean, "Std": std, "Min": min_val, "25%": p_25, "50%": p_50, "75%": p_75, "Max" : max_val
            }})

    return features


def print_features_table(features):
    labels = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    space = 15
    # print(header_row)
    print(''.ljust(space), end='\t')
    for feature in features:
        for key, value in feature.items():
            print(key.ljust(len(key) + 10), end='\t')
    print('\n')

    for label in labels:
        print(label.ljust(space), end='\t')
        for feature in features:
            for key, value in feature.items():
                print(f'{value[label]:.6f}'.ljust(len(key) + 10), end='\t')
        print('\n')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python describe.py <path to csv file>")
        sys.exit(1)
    
    # read the csv path from the command line
    path = sys.argv[1]
    # read the csv file
    headers, data = read_csv(path)
    # describe the data
    features = describe_data(headers, data)
    # print the table
    print_features_table(features)
