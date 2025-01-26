import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Step 1: Load the datasets
X = pd.read_csv('logisticX.csv').values  # Independent variables
y = pd.read_csv('logisticY.csv').values.flatten()  # Dependent variable (flatten to 1D array)

# Step 2: Add intercept term
X = np.c_[np.ones(X.shape[0]), X]  # Add a column of ones for the intercept

# Step 3: Initialize parameters
weights = np.zeros(X.shape[1])  # Initialize weights to zeros
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

# Train logistic regression model
final_weights, cost_history = gradient_descent(X, y, weights, learning_rate, iterations)

# Step 4: Make predictions
def predict(X, weights):
    probabilities = sigmoid(np.dot(X, weights))
    return [1 if prob >= 0.5 else 0 for prob in probabilities]

y_pred = predict(X, final_weights)

# Step 5: Calculate metrics
conf_matrix = confusion_matrix(y, y_pred)
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

# Step 6: Print results
print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Step 7: Optional - Plot the dataset and decision boundary for visualization
plt.figure(figsize=(8, 6))

# Separate points by class
class_0 = X[y == 0]
class_1 = X[y == 1]

# Plot class 0
plt.scatter(class_0[:, 1], class_0[:, 2], color='red', label='Class 0')
# Plot class 1
plt.scatter(class_1[:, 1], class_1[:, 2], color='blue', label='Class 1')

# Plot decision boundary
x_values = np.array([min(X[:, 1]), max(X[:, 1])])
y_values = -(final_weights[0] + final_weights[1] * x_values) / final_weights[2]
plt.plot(x_values, y_values, color='green', label='Decision Boundary')

# Aesthetics
plt.title("Dataset with Decision Boundary", fontsize=14)
plt.xlabel("Feature 1", fontsize=12)
plt.ylabel("Feature 2", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()