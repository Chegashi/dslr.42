
import csv

def read_csv(path):
    data = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            data.append(row)
    return headers, data

def is_numerical(column):
    for value in column:
        try:
            float(value)
            return True
        except ValueError:
            return False