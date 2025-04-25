#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from scipy.stats import pearsonr

def scatter_plot(path):
    try:
        data = pd.read_csv(path)
        
        features = data.select_dtypes(include=[np.number]).columns
        
        correlations = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                if features[i] != features[j]:
                    valid_data = data[[features[i], features[j]]].dropna()
                    if len(valid_data) > 0:
                        corr, _ = pearsonr(valid_data[features[i]], valid_data[features[j]])
                        correlations.append((features[i], features[j], abs(corr)))
        
        correlations.sort(key=lambda x: x[2], reverse=True)
        
        most_similar = correlations[0]
        feature1, feature2, correlation = most_similar
        
        print(f"The two most similar features are: {feature1} and {feature2}")
        print(f"With a correlation coefficient of: {correlation}")
        
        plt.figure(figsize=(10, 6))
        
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
        plt.savefig('similar_features.pdf')
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py <dataset_path>")
        sys.exit(1)
    
    path = sys.argv[1]
    scatter_plot(path)