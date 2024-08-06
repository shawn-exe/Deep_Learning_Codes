import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Sample data for binary classification (spam or not spam)
X = np.array([[0.1, 0.2], [0.4, 0.4], [0.5, 0.5], [0.9, 0.8]])
y = np.array([0, 0, 1, 1])  # Labels: 0 (not spam), 1 (spam)

# Define a simple model with sigmoid activation
model = models.Sequential()
model.add(layers.Dense(1, activation='sigmoid', input_shape=(2,)))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=1)

# Evaluate the model
loss, accuracy = model.evaluate(X, y)
print(f"Sigmoid Model Accuracy: {accuracy}")

# Make predictions
predictions = model.predict(X)

# Print predictions alongside actual labels
for i, prediction in enumerate(predictions):
    spam_or_not = "Spam" if prediction > 0.5 else "Not Spam"
    print(f"Input: {X[i]}, Predicted: {spam_or_not}, Actual: {'Spam' if y[i] == 1 else 'Not Spam'}")
