# What are the two features that are similar ?
# create a scatter plot visualizing the relationship between two features from a dataset based on the house’s 
# 78 pairs of features from the dataset
# Astronomy vs Defense Against the Dark Arts

import matplotlib.pyplot as plt
import pandas as pd
import itertools



def scatter_plot(path):
    data = pd.read_csv(path)

    features = data.columns[6:]
    houses = data['Hogwarts House'].unique()
    colors = ['blue', 'green', 'red', 'purple'] 
    # Creating pairs of features
    feature_pairs = list(itertools.combinations(features, 2))
    for pair in feature_pairs:
        plt.figure(figsize=(10, 6))
        # lets plot each house in a different color
        for house, color in zip(houses, colors):
            subset = data[data['Hogwarts House'] == house]
            plt.scatter(subset[pair[0]], subset[pair[1]], alpha=0.5, c=color, label=house)

        plt.title(f'{pair[0]} vs {pair[1]}')
        plt.xlabel(pair[0])
        plt.ylabel(pair[1])
        plt.show()

if __name__ == "__main__":
    path = 'csv/dataset_train.csv'
    scatter_plot(path)