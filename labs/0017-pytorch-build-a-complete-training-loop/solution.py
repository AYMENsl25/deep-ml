import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    """
    Train a PyTorch model and return training history.
    """

    history = []

    # 1. Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # 2. Epoch loop
    for epoch in range(epochs):

        # --------------------
        # Training
        # --------------------
        model.train()

        n = X_train.shape[0]

        # Shuffle data
        indices = torch.randperm(n)

        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        total_train_loss = 0.0

        # Mini-batches
        for start in range(0, n, batch_size):

            end = start + batch_size

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Clear old gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(X_batch)

            # Compute loss
            loss = loss_fn(logits, y_batch)

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            # Accumulate loss
            total_train_loss += loss.item() * len(X_batch)

        # Average loss over all training examples
        train_loss = total_train_loss / n

        # --------------------
        # Validation
        # --------------------
        model.eval()

        with torch.no_grad():

            val_logits = model(X_val)

            val_loss = loss_fn(val_logits, y_val).item()

            predictions = val_logits.argmax(dim=1)

            correct = (predictions == y_val).sum().item()

            val_accuracy = correct / len(y_val)

        # --------------------
        # Save epoch metrics
        # --------------------
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

    return history