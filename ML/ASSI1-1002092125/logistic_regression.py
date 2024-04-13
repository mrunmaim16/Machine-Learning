import numpy as np
import argparse

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_newton(X, y, learning_rate=0.01, n_iterations=1000, tol=1e-6):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(n_iterations):
        linear_model = np.dot(X, weights) + bias
        y_predicted = sigmoid(linear_model)

        # Compute gradient
        gradient_w = np.dot(X.T, (y - y_predicted)) / n_samples
        gradient_b = np.sum(y - y_predicted) / n_samples

        # Compute Hessian matrix
        W = y_predicted * (1 - y_predicted)
        Hessian_w = -np.dot(X.T, np.dot(np.diag(W), X)) / n_samples

        # Update parameters using Newton's method
        weights -= np.linalg.inv(Hessian_w).dot(gradient_w)
        bias += learning_rate * gradient_b

    linear_model = np.dot(X, weights) + bias
    y_predicted = sigmoid(linear_model)
    y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
    return np.array(y_predicted_cls)

def logistic_regression_gradient_ascent(X, y, learning_rate=0.001, n_iterations=1000, tol=1e-6):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(n_iterations):
        linear_model = np.dot(X, weights) + bias
        y_predicted = sigmoid(linear_model)

        # Compute gradient
        gradient_w = np.dot(X.T, (y - y_predicted)) / n_samples
        gradient_b = np.sum(y - y_predicted) / n_samples

        # Update parameters using gradient ascent
        weights += learning_rate * gradient_w
        bias += learning_rate * gradient_b

    linear_model = np.dot(X, weights) + bias
    y_predicted = sigmoid(linear_model)
    y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
    return np.array(y_predicted_cls)

def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = correct_predictions / len(y_true)
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistic Regression with Newton's Method or Gradient Ascent")
    parser.add_argument("--data", type=str, help="Path to data file (CSV format)")
    parser.add_argument("--test_data", type=str, help="Path to test data file (CSV format)")
    parser.add_argument("--method", type=str, default="newton", choices=["newton", "gradient"], help="Optimization method (newton or gradient)")
    args = parser.parse_args()

    if args.data and args.test_data:
        data = np.genfromtxt(args.data, delimiter=',')
        test_data = np.genfromtxt(args.test_data, delimiter=',')

        X = data[1:, :-1]
        y = data[1:, -1]
        X_test = test_data[1:, :-1]
        y_test = test_data[1:, -1]

        if args.method == "newton":
            # Run logistic regression with Newton's method
            predictions = logistic_regression_newton(X_test, y_test)
        else:
            # Run logistic regression with gradient ascent
            predictions = logistic_regression_gradient_ascent(X_test, y_test)

        print("Predictions:", predictions)
        test_accuracy = calculate_accuracy(y_test, predictions)
        print("Accuracy: {:.4f}".format(test_accuracy))
    else:
        print("Please provide the path to the training data file using the '--data' argument and testing data file using the '--test_data' argument.")
