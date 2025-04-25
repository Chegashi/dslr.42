#!/usr/bin/env python3
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys

def pair_plot(path):
    try:
        data = pd.read_csv(path)
        
        if 'Index' in data.columns:
            del data['Index']


        sns.pairplot(data, hue='Hogwarts House', palette=dict(Ravenclaw="Blue", Slytherin="Green", Hufflepuff="Yellow", Gryffindor="Red"))
        plt.savefig("pair_plot.png")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = '../ressources/datasets/dataset_train.csv'
    pair_plot(path)
