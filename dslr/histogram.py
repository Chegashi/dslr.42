#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import sys

#  Generate a histogram plot for a feature of a dataset
#  'Arithmancy', 'Care of Magical Creatures' are the one who have the most homogeneous score

def histogram(path):
    data = pd.read_csv(path)

    courses = data.columns[6:] # assuming that the course start from the 6th column
    houses = data['Hogwarts House'].unique()  # ['Ravenclaw' 'Slytherin' 'Gryffindor' 'Hufflepuff'] 

    most_homogeneous = None
    min_variance = float('inf')

    for course in courses:
        # each course will have its own separate histogram plot
        plt.figure(figsize=(10, 6))
        
        # Store variances for comparison
        house_variances = []
        
        for house in houses:
            score = data[data['Hogwarts House'] == house][course].dropna() # dropna() is used to remove the NaN value in the data
            # plot the histogram
            plt.hist(score, bins=20, alpha=0.5, label=house)
            # Calculate variance for homogeneity checking
            house_variances.append(score.var())
            
        # Calculate the variance between house variances to find homogeneity
        between_house_variance = pd.Series(house_variances).var()
        
        # Check if this is the most homogeneous
        if between_house_variance < min_variance:
            min_variance = between_house_variance
            most_homogeneous = course
            
        plt.title(f"{course} Distribution")
        plt.xlabel('Score')
        plt.ylabel("Number of students")
        plt.legend()
        plt.show()

    print(f"\nThe most homogeneous score distribution across all houses is in: {most_homogeneous}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python histogram.py <dataset_path>")
        sys.exit(1)
    
    path = sys.argv[1]
    histogram(path)

