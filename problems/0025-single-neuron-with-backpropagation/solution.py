import numpy as np

def train_neuron(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: float,
    learning_rate: float,
    epochs: int
) -> (np.ndarray, float, list[float]):

    # Convert inputs to NumPy arrays
    X = np.array(features, dtype=float)
    y = np.array(labels, dtype=float)

    weights = np.array(initial_weights, dtype=float)
    bias = float(initial_bias)

    mse_values = []

    for epoch in range(epochs):

        # -------------------------
        # Forward pass
        # -------------------------

        # Linear combination
        z = X @ weights + bias

        # Sigmoid activation
        predictions = 1 / (1 + np.exp(-z))

        # Error
        error = predictions - y

        # MSE loss
        mse = np.mean(error ** 2)

        # Store loss BEFORE updating parameters
        mse_values.append(mse)

        # -------------------------
        # Backward pass
        # -------------------------

        n = len(y)

        # Gradient through MSE + sigmoid
        grad_z = (
            (2 / n)
            * error
            * predictions
            * (1 - predictions)
        )

        # Weight gradient
        grad_weights = X.T @ grad_z

        # Bias gradient
        grad_bias = np.sum(grad_z)

        # -------------------------
        # Gradient descent update
        # -------------------------

        weights = weights - learning_rate * grad_weights

        bias = bias - learning_rate * grad_bias

    # -------------------------
    # Round results
    # -------------------------

    updated_weights = np.round(weights, 4)

    updated_bias = round(float(bias), 4)

    mse_values = [
        round(float(mse), 4)
        for mse in mse_values
    ]

    return updated_weights, updated_bias, mse_values