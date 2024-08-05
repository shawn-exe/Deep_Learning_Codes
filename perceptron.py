import numpy as np

# Define the perceptron function
def perceptron(x, w, b):
    # Compute the weighted sum
    z = np.dot(x, w) + b
    
    # Step activation function
    return 1 if z >= 0 else 0

# Given inputs and weights
x = np.array([0.5, 0.3])
w = np.array([0.4, -0.6])
b = 0.1

# Calculate the output of the perceptron
output = perceptron(x, w, b)
print(f"Perceptron Output: {output}")
