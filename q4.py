import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the datasets
# Replace 'X.csv' and 'y.csv' with the actual file paths if different
X = pd.read_csv('logisticX.csv').values  # Independent variables
y = pd.read_csv('logisticY.csv').values.flatten()  # Dependent variable (flatten to 1D array)

# Step 2: Introduce two new independent variables (squares of the original variables)
X_new = np.c_[X, X[:, 0]**2, X[:, 1]**2]  # Add squared features
X_new = np.c_[np.ones(X_new.shape[0]), X_new]  # Add a column of ones for the intercept

# Step 3: Initialize parameters
weights = np.zeros(X_new.shape[1])  # Initialize weights to zeros
learning_rate = 0.1  # Learning rate
iterations = 1000  # Number of iterations

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define cost function
def compute_cost(X, y, weights):
    m = len(y)
    predictions = sigmoid(np.dot(X, weights))
    epsilon = 1e-15  # To prevent log(0)
    cost = -1/m * np.sum(y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon))
    return cost

# Gradient Descent Function
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

# Train logistic regression model on the new dataset
final_weights, cost_history = gradient_descent(X_new, y, weights, learning_rate, iterations)

# Step 4: Plot the dataset and decision boundary
plt.figure(figsize=(8, 6))

# Separate points by class
class_0 = X[y == 0]
class_1 = X[y == 1]

# Plot class 0
plt.scatter(class_0[:, 0], class_0[:, 1], color='red', label='Class 0')
# Plot class 1
plt.scatter(class_1[:, 0], class_1[:, 1], color='blue', label='Class 1')

# Plot decision boundary
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid = np.c_[xx.ravel(), yy.ravel(), xx.ravel()**2, yy.ravel()**2]
grid = np.c_[np.ones(grid.shape[0]), grid]  # Add intercept term
predictions = sigmoid(np.dot(grid, final_weights)).reshape(xx.shape)

# Contour plot for the decision boundary
plt.contour(xx, yy, predictions, levels=[0.5], colors='green', linewidths=2, label='Decision Boundary')

# Aesthetics
plt.title("Dataset with New Features and Decision Boundary", fontsize=14)
plt.xlabel("Feature 1", fontsize=12)
plt.ylabel("Feature 2", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# Step 5: Print final cost and weights
final_cost = cost_history[-1]
print(f"Final Cost Function Value: {final_cost}")
print(f"Final Weights (Coefficients): {final_weights}")