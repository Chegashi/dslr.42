#!/usr/bin/env python3
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys

# These features apparently do not provide enough differentiation between the houses and therefore are less useful for classification purposes in logistic regression.
# Arithmancy, Potions and Care of Magical Creatures are homogeneous features so they should be ignored

# Features that extract 1 house from all of the others should be selected since the logistic regression calculate weights for the houses one by one
# Divination, Muggle Studies, History of Magic, Transfiguration, Charms and Flying should be good features to use in the logistic regression

# displays a pair plot or scatter plot matrix

def pair_plot(path):
    try:
        data = pd.read_csv(path)
        
        if 'Index' in data.columns:
            del data['Index']
        # # similaire result (see Histogram)
        # del data['Care of Magical Creatures']
        # del data['Arithmancy']
        # # data identic with defense ag. (see scatter_plot)
        # del data['Astronomy']

        sns.pairplot(data, hue='Hogwarts House', palette=dict(Ravenclaw="Blue", Slytherin="Green", Hufflepuff="Yellow", Gryffindor="Red"))  # Assuming 'Hogwarts House' is a categorical feature
        plt.show()
    except Exception as e:
        print(e)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = '../ressources/datasets/dataset_train.csv'
    pair_plot(path)