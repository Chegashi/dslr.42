import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class LogisticRegressionGradientDescent:
    iterations: int = 1000
    learning_rate: float = 0.1
    weights: dict = {}
    X_mean: pd.Series
    X_std: pd.Series
    X_raw: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    houses: np.ndarray

    def __init__(self, path: str, iterations: int, learning_rate: float):
        """
        Initialize the logistic regression model with training data
        :param path: Path to the training dataset
        :param iterations: Number of iterations for gradient descent
        :param learning_rate: Learning rate for gradient descent
        :param plot: Whether to plot the cost function
        """
        self.iterations = iterations
        self.learning_rate = learning_rate
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
            np.dot(y, np.log(h + epsilon)) + np.dot((1 - y), np.log(1 - h + epsilon))
        )
        return cost

    def run_gradient_descent(self, X, y, theta, learning_rate, iterations):
        """Run gradient descent for logistic regression"""
        m = len(y)
        cost_history = []
        iteration_list = []

        # Create figure for real-time cost plotting
        plt.figure(figsize=(10, 6))
        plt.title("Cost Function During Training")
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.grid(True)

        for i in range(iterations):
            # Predict
            h = self.predict(X, theta)

            # Calculate gradient
            gradient = np.dot(X.T, (h - y)) / m

            # Update parameters
            theta = theta - learning_rate * gradient

            # Calculate and store cost at intervals
            if i % 10 == 0:  # More frequent updates for smoother plot
                cost = self.compute_cost(X, y, theta)
                cost_history.append(cost)
                iteration_list.append(i)

                # Real-time plot update (every 50 iterations to avoid slowing down)
                if i % 50 == 0:
                    print(f"Iteration {i}, Cost: {cost:.6f}")
                    plt.plot(iteration_list, cost_history, "b-")
                    plt.pause(0.01)

        plt.close()
        return theta, cost_history, iteration_list

    def train_one_vs_all(self):
        """Train one-vs-all logistic regression models"""
        self.weights = {}
        self.cost_histories = {}
        self.iteration_lists = {}
        learning_rate = self.learning_rate
        iterations = self.iterations

        for house in self.houses:
            print(f"\nTraining model for {house}...")

            # Create binary labels (1 for current house, 0 for others)
            y_binary = (self.y == house).astype(int).values

            # Initialize weights with zeros
            theta = np.zeros(self.X.shape[1])

            # Run gradient descent
            theta, cost_history, iteration_list = self.run_gradient_descent(
                self.X.values, y_binary, theta, learning_rate, iterations
            )

            # Store weights and training history for this house
            self.weights[house] = theta.tolist()
            self.cost_histories[house] = cost_history
            self.iteration_lists[house] = iteration_list
            print(f"Final cost for {house}: {cost_history[-1]:.6f}")

        # After training all models, plot comparisons
        self.plot_cost_comparison()
        self.plot_feature_importance()

    def plot_cost_comparison(self):
        """Compare cost function convergence for all houses"""
        plt.figure(figsize=(12, 8))

        for house in self.houses:
            plt.plot(
                self.iteration_lists[house],
                self.cost_histories[house],
                label=f"{house}",
            )

        plt.title("Cost Function Convergence by House")
        plt.xlabel("Iterations")
        plt.ylabel("Cost (Cross-Entropy Loss)")
        plt.legend()
        plt.grid(True)
        plt.savefig("cost_comparison.png")
        plt.show()
        print("Cost comparison plot saved as cost_comparison.png")

    def run_gradient_descent(self, X, y, theta, learning_rate, iterations):
        """Run gradient descent for logistic regression"""
        m = len(y)
        cost_history = []
        iteration_list = []

        # Create figure for real-time cost plotting
        plt.figure(figsize=(10, 6))
        plt.title("Cost Function During Training")
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.grid(True)

        for i in range(iterations):
            # Predict
            h = self.predict(X, theta)

            # Calculate gradient
            gradient = np.dot(X.T, (h - y)) / m

            # Update parameters
            theta = theta - learning_rate * gradient

            # Calculate and store cost at intervals
            if i % 10 == 0:  # More frequent updates for smoother plot
                cost = self.compute_cost(X, y, theta)
                cost_history.append(cost)
                iteration_list.append(i)

                # Real-time plot update (every 50 iterations to avoid slowing down)
                if i % 50 == 0:
                    print(f"Iteration {i}, Cost: {cost:.6f}")
                    plt.plot(iteration_list, cost_history, "b-")
                    plt.pause(0.01)

        plt.close()
        return theta, cost_history, iteration_list

    def train_one_vs_all(self):
        """Train one-vs-all logistic regression models"""
        self.weights = {}
        self.cost_histories = {}
        self.iteration_lists = {}
        learning_rate = self.learning_rate
        iterations = self.iterations

        for house in self.houses:
            print(f"\nTraining model for {house}...")

            # Create binary labels (1 for current house, 0 for others)
            y_binary = (self.y == house).astype(int).values

            # Initialize weights with zeros
            theta = np.zeros(self.X.shape[1])

            # Run gradient descent
            theta, cost_history, iteration_list = self.run_gradient_descent(
                self.X.values, y_binary, theta, learning_rate, iterations
            )

            # Store weights and training history for this house
            self.weights[house] = theta.tolist()
            self.cost_histories[house] = cost_history
            self.iteration_lists[house] = iteration_list
            print(f"Final cost for {house}: {cost_history[-1]:.6f}")

        # After training all models, plot comparisons
        self.plot_cost_comparison()
        self.plot_feature_importance()

    def plot_cost_comparison(self):
        """Compare cost function convergence for all houses"""
        plt.figure(figsize=(12, 8))

        for house in self.houses:
            plt.plot(
                self.iteration_lists[house],
                self.cost_histories[house],
                label=f"{house}",
            )

        plt.title("Cost Function Convergence by House")
        plt.xlabel("Iterations")
        plt.ylabel("Cost (Cross-Entropy Loss)")
        plt.legend()
        plt.grid(True)
        plt.savefig("cost_comparison.png")
        plt.show()
        print("Cost comparison plot saved as cost_comparison.png")

    def plot_feature_importance(self):
        """Visualize the importance of each feature for each house using multiple approaches"""
        feature_names = self.X.columns[1:]  # Skip bias term

        # 1. Create a feature weights heatmap for all houses
        self.plot_feature_heatmap(feature_names)

        # 2. Create radar charts for feature patterns
        self.plot_feature_radar(feature_names)

        # 3. Create a 2D projection of feature importance
        self.plot_feature_projection(feature_names)

    def plot_feature_heatmap(self, feature_names):
        """Create a heatmap of feature weights for all houses"""
        # Extract weights for all houses
        weights_array = np.zeros((len(self.houses), len(feature_names)))

        for i, house in enumerate(self.houses):
            # Skip bias term (index 0)
            weights_array[i] = np.array(self.weights[house][1:])

        plt.figure(figsize=(12, 8))

        # Create heatmap
        im = plt.imshow(weights_array, cmap="coolwarm")

        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label("Weight Value")

        # Set labels
        plt.yticks(range(len(self.houses)), self.houses)
        plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha="right")

        # Add grid
        plt.grid(False)

        plt.title("Feature Importance Heatmap by House")
        plt.tight_layout()
        plt.savefig("feature_heatmap.png")
        plt.show()
        print("Feature heatmap saved as feature_heatmap.png")

    def plot_feature_radar(self, feature_names):
        """Create radar charts showing feature patterns for each house"""
        # Set up figure
        fig = plt.figure(figsize=(15, 12))

        # Calculate number of variables
        N = len(feature_names)

        # Create angle for each feature
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop

        # Create subplots (2x2 grid)
        axes = [
            fig.add_subplot(2, 2, i + 1, polar=True) for i in range(len(self.houses))
        ]

        # Add feature names to the chart
        plt.subplots_adjust(hspace=0.4)

        # Plot each house
        for i, (ax, house) in enumerate(zip(axes, self.houses)):
            # Get weights and normalize them for better visualization
            weights = np.array(self.weights[house][1:])
            # Scale weights to [0,1] for better radar visualization
            min_weight = min(weights)
            max_weight = max(weights)
            if max_weight != min_weight:
                scaled_weights = (weights - min_weight) / (max_weight - min_weight)
            else:
                scaled_weights = np.zeros_like(weights)

            # Close the loop by adding the first value again
            values = list(scaled_weights) + [scaled_weights[0]]

            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle="solid", label=house)
            ax.fill(angles, values, alpha=0.25)

            # Set feature labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(feature_names, fontsize=8)

            # Remove y labels and set y limits
            ax.set_yticklabels([])
            ax.set_ylim(0, 1)

            # Add title
            ax.set_title(f"{house} Feature Pattern", size=11, y=1.1)

        plt.tight_layout()
        plt.savefig("feature_radar.png")
        plt.show()
        print("Feature radar charts saved as feature_radar.png")

    def plot_feature_projection(self, feature_names):
        """Create a 2D scatter plot of features based on their importance across houses"""
        from sklearn.decomposition import PCA

        # Extract weights for all houses and create a feature-by-house matrix
        feature_importance = np.zeros((len(feature_names), len(self.houses)))

        for i, house in enumerate(self.houses):
            feature_importance[:, i] = np.array(self.weights[house][1:])

        # Apply PCA to reduce to 2 dimensions
        pca = PCA(n_components=2)
        feature_coords = pca.fit_transform(feature_importance)

        # Create scatter plot
        plt.figure(figsize=(12, 10))

        # Plot points
        scatter = plt.scatter(
            feature_coords[:, 0],
            feature_coords[:, 1],
            c=np.sum(np.abs(feature_importance), axis=1),
            cmap="viridis",
            s=100,
            alpha=0.7,
        )

        # Add feature labels
        for i, feature in enumerate(feature_names):
            plt.annotate(
                feature,
                (feature_coords[i, 0], feature_coords[i, 1]),
                fontsize=9,
                ha="center",
            )

        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label("Overall Importance (Sum of Absolute Weights)")

        # Add title and labels
        plt.title("2D Projection of Features by Importance Across Houses")
        plt.xlabel(
            f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)"
        )
        plt.ylabel(
            f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)"
        )

        # Add grid
        plt.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig("feature_projection.png")
        plt.show()
        print("Feature projection plot saved as feature_projection.png")

    def save_model_weights(self):
        """Save model parameters to a file"""
        model = {
            "weights": self.weights,
            "X_mean": self.X_mean.to_dict(),
            "X_std": self.X_std.to_dict(),
        }

        with open(f"model_weights.json", "w") as f:
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
