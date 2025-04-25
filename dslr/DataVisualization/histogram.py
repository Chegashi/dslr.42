#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import sys

def histogram(path):
    data = pd.read_csv(path)

    courses = data.columns[6:]
    houses = data['Hogwarts House'].unique()

    most_homogeneous = None
    min_variance = float('inf')

    for course in courses:
        plt.figure(figsize=(10, 6))
        
        house_variances = []
        
        for house in houses:
            score = data[data['Hogwarts House'] == house][course].dropna()
            plt.hist(score, bins=20, alpha=0.5, label=house)
            house_variances.append(score.var())
            
        between_house_variance = pd.Series(house_variances).var()
        
        if between_house_variance < min_variance:
            min_variance = between_house_variance
            most_homogeneous = course
            
        plt.title(f"{course} Distribution")
        plt.xlabel('Score')
        plt.ylabel("Number of students")
        plt.legend()
        plt.savefig(f"{course}.png")

    print(f"\nThe most homogeneous score distribution across all houses is in: {most_homogeneous}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python histogram.py <dataset_path>")
        sys.exit(1)
    
    path = sys.argv[1]
    histogram(path)

