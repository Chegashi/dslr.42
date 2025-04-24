import argparse
import json

import numpy as np
import pandas as pd


class LogisticRegressionPredictor:
    def __init__(self, test_path: str, weights_path: str):
        print("Loading model weights...")
        self.load_model(weights_path)

        print("Loading test dataset...")
        self.load_test_data(test_path)

        print("Making predictions...")
        self.predict()

        print("Saving predictions to houses.csv...")
        self.save_predictions()

        print("Prediction complete!")

    def load_model(self, weights_path: str):
        """Load the trained model parameters"""
        with open(weights_path, "r") as f:
            model = json.load(f)

        self.weights = model["weights"]
        self.X_mean = model["X_mean"]
        self.X_std = model["X_std"]

    def load_test_data(self, test_path: str):
        """Load and preprocess the test data"""
        self.test_df = pd.read_csv(test_path)

        # Extract feature columns
        feature_columns = list(self.X_mean.keys())
        self.X_test_raw = self.test_df[feature_columns]

        # Handle missing values
        self.X_test_raw = self.X_test_raw.fillna(self.X_test_raw.mean())

        # Normalize features using training data parameters
        self.X_test = pd.DataFrame()
        for col in feature_columns:
            self.X_test[col] = (self.X_test_raw[col] - self.X_mean[col]) / self.X_std[
                col
            ]

        # Add bias term
        self.X_test.insert(0, "bias", 1)

    def sigmoid(self, z):
        """Sigmoid function implementation"""
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self):
        """Make predictions for the test data"""
        self.predictions = []
        houses = list(self.weights.keys())

        for _, student in self.X_test.iterrows():
            probabilities = {}

            for house in houses:
                theta = np.array(self.weights[house])
                # Calculate probability using sigmoid
                probability = self.sigmoid(np.dot(student, theta))
                probabilities[house] = probability

            # Select house with highest probability
            predicted_house = max(probabilities, key=probabilities.get)
            self.predictions.append(predicted_house)

    def save_predictions(self):
        """Save predictions to a CSV file"""
        predictions_df = pd.DataFrame(
            {"Index": self.test_df["Index"], "Hogwarts House": self.predictions}
        )

        predictions_df.to_csv("houses.csv", index=False)


def parse_data(path: str) -> str:
    """Validate the input data file"""
    try:
        open(path, "r")
    except FileNotFoundError:
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    if not path:
        raise argparse.ArgumentTypeError("File path cannot be empty.")
    if not path.endswith(".csv") and not path.endswith(".json"):
        raise argparse.ArgumentTypeError("File must be a CSV or JSON file.")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=parse_data,
        help="Path to the dataset_test.csv file.",
    )
    parser.add_argument(
        "--weights",
        type=parse_data,
        help="Path to the model weights file.",
        default="model_weights.json",
    )
    args = parser.parse_args()
    try:
        LogisticRegressionPredictor(args.dataset, args.weights)
    except ValueError as e:
        print(e)
