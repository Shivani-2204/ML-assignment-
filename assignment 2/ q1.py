import numpy as np
import pandas as pd

# Load data
X = pd.read_csv('logisticX.csv').values
y = pd.read_csv('logisticY.csv').values.flatten()

# Add intercept term
X = np.c_[np.ones(X.shape[0]), X]

# Initialize parameters
weights = np.zeros(X.shape[1])
learning_rate = 0.1
iterations = 1000

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define cost function
def compute_cost(X, y, weights):
    m = len(y)
    predictions = sigmoid(np.dot(X, weights))
    epsilon = 1e-15
    cost = -1/m * np.sum(y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon))
    return cost

# Gradient Descent
def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    cost_history = []
    for _ in range(iterations):
        predictions = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (predictions - y)) / m
        weights -= learning_rate * gradient
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)
    return weights, cost_history

# Train the logistic regression model
final_weights, cost_history = gradient_descent(X, y, weights, learning_rate, iterations)

# Final results
final_cost = cost_history[-1]
print(f"Final Cost Function Value: {final_cost}")
print(f"Final Weights (Coefficients): {final_weights}")
 
