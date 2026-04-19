import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
def generate_dummy_data(samples=100):
    X = np.random.rand(samples, 5)  # 5 features
    y = X.sum(axis=1, keepdims=True) + np.random.randn(samples,1)*0.1
    return X, y

# Ai generated ploting
def plot_predictions(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    n = len(y_true)

    plt.figure(figsize=(10,6))
    plt.scatter(range(n), y_true, color='blue', label='True', s=40)
    plt.scatter(range(n), y_pred, color='red', label='Predicted', s=40)

    for i in range(n):
        plt.arrow(i, y_true[i], 0, y_pred[i]-y_true[i],
                  color='gray', alpha=0.5,
                  head_width=0.2, head_length=0.1, length_includes_head=True)

    plt.xlabel('Sample index')
    plt.ylabel('Value')
    plt.title('True vs Predicted Values with Arrows')
    plt.legend()
    plt.grid(True)
    plt.show()