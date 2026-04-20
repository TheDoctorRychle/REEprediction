"""
Training loop for the MLP model.
Returns the loss history (useful for the learning curve).
"""


def train_model(model, X, y, epochs=100, lr=0.01, verbose=False):
    """
    Trains the MLP model for the given number of epochs.

    Arguments:
      model   -- MLP object
      X       -- training feature matrix
      y       -- target vector
      epochs  -- number of epochs
      lr      -- learning rate
      verbose -- whether to print loss every 50 epochs

    Returns:
      list of MSE losses for each epoch (useful for learning curve plot)
    """
    loss_history = []

    for epoch in range(epochs):
        # Forward propagation
        y_pred = model.forward_propagation(X)

        # MSE loss
        loss = ((y_pred - y) ** 2).mean()
        loss_history.append(float(loss))

        # Backward propagation and weight update
        grads_W, grads_b = model.backward_propagation(y)
        model.update(grads_W, grads_b, lr)

        if verbose and epoch % 50 == 0:
            print(f"  Epoch {epoch:>4d}/{epochs}  Loss: {loss:.6f}")

    return loss_history
