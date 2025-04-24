#!/usr/bin/env python3
# What are the two features that are similar?
# Create a scatter plot visualizing the relationship between features from a dataset
# to identify the two most similar features

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from scipy.stats import pearsonr

def scatter_plot(path):
    """
    Display scatter plots to identify the two most similar features.
    Similarity is measured using Pearson correlation coefficient.
    """
    try:
        # Load the data
        data = pd.read_csv(path)
        
        # Get numerical features (skip non-numeric columns)
        features = data.select_dtypes(include=[np.number]).columns
        
        # Calculate correlation between all feature pairs
        correlations = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                if features[i] != features[j]:
                    # Clean data by removing NaN values
                    valid_data = data[[features[i], features[j]]].dropna()
                    if len(valid_data) > 0:
                        corr, _ = pearsonr(valid_data[features[i]], valid_data[features[j]])
                        # Store absolute correlation to find most similar regardless of direction
                        correlations.append((features[i], features[j], abs(corr)))
        
        # Sort by correlation (highest first)
        correlations.sort(key=lambda x: x[2], reverse=True)
        
        # Get the most similar pair
        most_similar = correlations[0]
        feature1, feature2, correlation = most_similar
        
        print(f"The two most similar features are: {feature1} and {feature2}")
        print(f"With a correlation coefficient of: {correlation}")
        
        # Plot the most similar pair
        plt.figure(figsize=(10, 6))
        
        # If Hogwarts House column exists, color by house
        if 'Hogwarts House' in data.columns:
            houses = data['Hogwarts House'].unique()
            colors = ['blue', 'green', 'red', 'purple']
            
            for house, color in zip(houses, colors):
                subset = data[data['Hogwarts House'] == house]
                plt.scatter(subset[feature1], subset[feature2], alpha=0.5, c=color, label=house)
            
            plt.legend()
        else:
            plt.scatter(data[feature1], data[feature2], alpha=0.5)
        
        plt.title(f'Most Similar Features: {feature1} vs {feature2} (r={correlation:.2f})')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.savefig('similar_features.png')
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py <dataset_path>")
        sys.exit(1)
    
    path = sys.argv[1]
    scatter_plot(path)