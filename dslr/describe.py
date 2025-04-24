#!/usr/bin/env python3
import sys
import math
from utils import is_numerical, read_csv


def describe_data(headers, data):
    features = []
    for col_idx, column in enumerate(zip(*data)):
        if is_numerical(column):
            # convert to numeric values and filter out non-numeric
            numeric_values = [float(value) for value in column if value.replace('.', '', 1).isdigit()]
            
            # Count - manually count elements
            count = 0
            for _ in numeric_values:
                count += 1
                
            if count == 0:
                continue
            
            # Mean - sum divided by count (manual implementation)
            total = 0
            for val in numeric_values:
                total += val
            mean = total / count
            
            # Standard deviation - manual implementation
            sum_squared_diff = 0
            for val in numeric_values:
                sum_squared_diff += (val - mean) ** 2
            std = (sum_squared_diff / count) ** 0.5
            
            # Min and Max - manual implementation
            min_val = numeric_values[0]
            max_val = numeric_values[0]
            for val in numeric_values:
                if val < min_val:
                    min_val = val
                if val > max_val:
                    max_val = val
            
            # Sort values for percentiles - manual implementation
            sorted_values = numeric_values.copy()
            # Simple bubble sort - use different variable names to avoid conflict
            for i_sort in range(count):
                for j_sort in range(0, count - i_sort - 1):
                    if sorted_values[j_sort] > sorted_values[j_sort + 1]:
                        sorted_values[j_sort], sorted_values[j_sort + 1] = sorted_values[j_sort + 1], sorted_values[j_sort]
            
            # Calculate percentiles - manual implementation
            p_25 = sorted_values[int(count * 0.25)]  # Q1
            p_50 = sorted_values[int(count * 0.50)]  # Q2/median
            p_75 = sorted_values[int(count * 0.75)]  # Q3
            
            # Additional percentiles
            q1 = sorted_values[int(count * 0.1)]
            q2 = sorted_values[int(count * 0.2)]
            q3 = sorted_values[int(count * 0.3)]
            q4 = sorted_values[int(count * 0.4)]
            q6 = sorted_values[int(count * 0.6)]
            q7 = sorted_values[int(count * 0.7)]
            q8 = sorted_values[int(count * 0.8)]
            q9 = sorted_values[int(count * 0.9)]
            
            # Additional statistics - manual implementation
            range_val = max_val - min_val
            variance = std ** 2
            iqr = p_75 - p_25
            
            # Coefficient of variation - manual implementation
            cv = (std / mean) * 100 if mean != 0 else float('nan')
            
            features.append({headers[col_idx]: {
                "Count": count, 
                "Mean": mean, 
                "Std": std, 
                "Var": variance, 
                "Min": min_val, 
                "10%": q1,
                "20%": q2,
                "25%": p_25, 
                "30%": q3,
                "40%": q4,
                "50%": p_50, 
                "60%": q6,
                "70%": q7,
                "75%": p_75, 
                "80%": q8,
                "90%": q9,
                "Max": max_val,
                "Range": range_val,
                "IQR": iqr,
                "CV%": cv
            }})

    return features


def print_features_table(features):
    # Updated list of labels to include all statistics
    labels = ['Count', 'Mean', 'Std', 'Var', 'Min', '10%', '20%', '25%', '30%', '40%', 
              '50%', '60%', '70%', '75%', '80%', '90%', 'Max', 'Range', 'IQR', 'CV%']
    space = 15
    
    # Print feature names
    print(''.ljust(space), end='\t')
    for feature in features:
        for key, value in feature.items():
            print(key.ljust(len(key) + 10), end='\t')
    print('\n')

    # Print all statistics for each feature
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
