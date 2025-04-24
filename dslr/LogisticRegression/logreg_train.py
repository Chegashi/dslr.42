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
        # Select features and target
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

        # Extract features and target
        self.X_raw = self.data[feature_columns]
        self.y = self.data["Hogwarts House"]

        # Handle missing values
        self.X_raw = self.X_raw.fillna(self.X_raw.mean())

        # Store means and standard deviations for later use
        self.X_mean = self.X_raw.mean()
        self.X_std = self.X_raw.std()

        # Normalize features
        self.X = (self.X_raw - self.X_mean) / self.X_std

        # Add bias term (intercept)
        self.X.insert(0, "bias", 1)

        # Get unique houses
        self.houses = self.y.unique()
        print(f"Found houses: {self.houses}")

    def sigmoid(self, z):
        """Sigmoid function implementation"""
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, X, theta):
        """Predict using logistic regression"""
        return self.sigmoid(np.dot(X, theta))

    def compute_cost(self, X, y, theta):
        """Compute cost for logistic regression"""
        m = len(y)
        h = self.predict(X, theta)
        # Avoid log(0) errors
        epsilon = 1e-5
        cost = (-1 / m) * (
            np.dot(y, np.log(h + epsilon))
            + np.dot((1 - y), np.log(1 - h + epsilon))
        )
        return cost

    def run_gradient_descent(self, X, y, theta):
        """Run gradient descent for logistic regression"""
        m = len(y)
        costs = []

        for i in range(self.iterations):
            # Predict
            h = self.predict(X, theta)

            # Calculate gradient
            gradient = np.dot(X.T, (h - y)) / m

            # Update parameters
            theta = theta - self.learning_rate * gradient

            # Calculate cost and store it
            if i % 100 == 0:
                cost = self.compute_cost(X, y, theta)
                costs.append(cost)
                print(f"Iteration {i}, Cost: {cost}")

        return theta, costs

    def train_one_vs_all(self):
        """Train one-vs-all logistic regression models"""
        self.weights = {}

        for house in self.houses:
            print(f"\nTraining model for {house}...")

            # Create binary labels (1 for current house, 0 for others)
            y_binary = (self.y == house).astype(int).values

            # Initialize weights with zeros
            theta = np.zeros(self.X.shape[1])

            # Run gradient descent
            theta, costs = self.run_gradient_descent(
                self.X.values, y_binary, theta
            )

            # Store weights for this house
            self.weights[house] = theta.tolist()
            print(f"Final cost for {house}: {costs[-1]}")

    def save_model_weights(self):
        """Save model parameters to a file"""
        model = {
            "weights": self.weights,
            "X_mean": self.X_mean.to_dict(),
            "X_std": self.X_std.to_dict(),
        }

        with open("model_weights.json", "w") as f:
            json.dump(model, f)


def parse_data(path: str) -> str:
    """Validate the input data file"""
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
