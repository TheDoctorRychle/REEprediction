def train_model(model, X, y, epochs=100, lr=0.01):
    for epoch in range(epochs):
        y_pred = model.forward_propagation(X)
        loss = ((y_pred - y)**2).mean()
        grads_W, grads_b = model.backward_propagation(y)
        model.update(grads_W, grads_b, lr)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")