import argparse
import json

import numpy as np
import pandas as pd


class LogisticRegressionGradientDescent:

    iterations: int
    learning_rate: float

    def __init__(
        self, path: str, iterations: int = 1000, learning_rate: float = 0.1
    ):
        self.learning_rate = learning_rate
        self.iterations = iterations

        print("Loading training dataset...")
        self.data = pd.read_csv(path)
        self.process_data()
        self.train_one_vs_all()
        self.save_model_weights()
        print("Training complete. Model weights saved to model_weights.json")

    def process_data(self):
        feature_columns = [
            "Arithmancy",
            "Astronomy",
            "Herbology",
            "Defense Against the Dark Arts",
            "Divination",
            "Muggle Studies",
            "Ancient Runes",
            "History of Magic",
            "Transfiguration",
            "Potions",
            "Care of Magical Creatures",
            "Charms",
            "Flying",
        ]

        self.X_raw = self.data[feature_columns]
        self.y = self.data["Hogwarts House"]

        self.X_raw = self.X_raw.fillna(self.X_raw.mean())

        self.X_mean = self.X_raw.mean()
        self.X_std = self.X_raw.std()

        self.X = (self.X_raw - self.X_mean) / self.X_std

        self.X.insert(0, "bias", 1)

        self.houses = self.y.unique()
        print(f"Found houses: {self.houses}")

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, X, theta):
        return self.sigmoid(np.dot(X, theta))

    def compute_cost(self, X, y, theta):
        m = len(y)
        h = self.predict(X, theta)
        epsilon = 1e-5
        cost = (-1 / m) * (
            np.dot(y, np.log(h + epsilon))
            + np.dot((1 - y), np.log(1 - h + epsilon))
        )
        return cost

    def run_gradient_descent(self, X, y, theta):
        m = len(y)
        costs = []

        for i in range(self.iterations):
            h = self.predict(X, theta)

            gradient = np.dot(X.T, (h - y)) / m

            theta = theta - self.learning_rate * gradient

            if i % 100 == 0:
                cost = self.compute_cost(X, y, theta)
                costs.append(cost)
                print(f"Iteration {i}, Cost: {cost}")

        return theta, costs

    def train_one_vs_all(self):
        self.weights = {}

        for house in self.houses:
            print(f"\nTraining model for {house}...")

            y_binary = (self.y == house).astype(int).values

            theta = np.zeros(self.X.shape[1])

            theta, costs = self.run_gradient_descent(
                self.X.values, y_binary, theta
            )

            self.weights[house] = theta.tolist()
            print(f"Final cost for {house}: {costs[-1]}")

    def save_model_weights(self):
        model = {
            "weights": self.weights,
            "X_mean": self.X_mean.to_dict(),
            "X_std": self.X_std.to_dict(),
        }

        with open("model_weights.json", "w") as f:
            json.dump(model, f)


def parse_data(path: str) -> str:
    try:
        open(path, "r")
    except FileNotFoundError:
        raise argparse.ArgumentTypeError("File not found.")
    if not path:
        raise argparse.ArgumentTypeError("File path cannot be empty.")
    if not path.endswith(".csv"):
        raise argparse.ArgumentTypeError("File must be a CSV file.")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=parse_data,
        required=True,
        help="Path to the dataset_train.csv file.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        required=False,
        help="Number of iterations for gradient descent.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        required=False,
        help="Learning rate for gradient descent.",
    )

    args = parser.parse_args()
    try:
        LogisticRegressionGradientDescent(
            args.dataset, args.iterations, args.learning_rate
        )
    except ValueError as e:
        print(e)
