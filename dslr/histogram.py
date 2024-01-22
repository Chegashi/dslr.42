import matplotlib.pyplot as plt
import pandas as pd

#  Generate a histogram plot for a feature of a dataset
#  'Arithmancy', 'Care of Magical Creatures' are the one who have the most homogeneous score

def histogram(path):
    data = pd.read_csv(path)

    courses = data.columns[6:] # assuming that the course start from the 6th column
    houses = data['Hogwarts House'].unique()  # ['Ravenclaw' 'Slytherin' 'Gryffindor' 'Hufflepuff'] 

    for course in courses:
        # each course will have its own separate histogram plot
        plt.figure(figsize=(10, 6))
        for house in houses:
            score = data[data['Hogwarts House'] == house][course].dropna() # dropna() is used to remove the NaN value in the data
            # plot the histogram
            plt.hist(score, bins=20, alpha=0.5, label=house)
        plt.title(course)
        plt.xlabel('Score')
        plt.ylabel("Number of student")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    path = 'csv/dataset_train.csv'
    histogram(path)

