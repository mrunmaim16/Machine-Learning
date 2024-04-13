import numpy as np
import argparse
# Import the `pyplot` module from the `matplotlib` library to plot the graph for visualization 
import matplotlib.pyplot as plt

class LinearRegressionGradientDescent:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            cost = (1 / n_samples) * np.sum((linear_model - y) ** 2)

            # Compute gradients
            weights_gradient = (2 / n_samples) * np.dot(X.T, (linear_model - y))
            bias_gradient = (2 / n_samples) * np.sum(linear_model - y)

            # Update parameters
            self.weights = self.weights- self.learning_rate * weights_gradient
            self.bias = self.bias - self.learning_rate * bias_gradient

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear Regression with Gradient Descent")
    parser.add_argument("--data", type=str, help="Path to data file (CSV format)")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for gradient descent")
    parser.add_argument("--n_iterations", type=int, default=1000, help="Number of iterations for gradient descent")
    args = parser.parse_args()

    if args.data:
        # Load data
        data = np.genfromtxt(args.data, delimiter=',')
        X = data[1:24, :-1]
        y = data[1:24, -1]
        X_test = data[25:, :-1]
        # Initialize and train the model
        model = LinearRegressionGradientDescent(learning_rate=args.learning_rate, n_iterations=args.n_iterations)
        model.fit(X, y)

        # Make predictions
        predictions = model.predict(X_test)

        print("Predictions:", predictions)
    else:
        print("Please provide the path to the data file using the '--data' argument.")

# Plot the linear graph with actual data points, linear regression line, connections, and predicted values

# Scatter plot for actual data points in blue
plt.scatter(X, y, color='blue', label='Actual data')

# Plot the linear regression line in red
plt.plot(X_test, predictions, color='red', label='Linear Regression')

# Connect the last actual data point to the first predicted value with a dashed green line
connection_line_x = [X[-1], X_test[0]]
connection_line_y = [y[-1], predictions[0]]
plt.plot(connection_line_x, connection_line_y, linestyle='--', color='green', label='Connection')

# Highlight predicted values with orange dots
plt.scatter(X_test, predictions, color='orange', label='Predicted values')

# Set plot title and axis labels
plt.title('Linear Regression with Connection and Predicted Values')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Display legend
plt.legend()

# Show the linear graph
plt.show()

