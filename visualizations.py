import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y):

    if X.shape[1] != 2:
        raise ValueError("plot_decision_boundary is only applicable for 2D datasets.")

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    predictions = model.forward(grid, training=False)
    predictions = np.argmax(predictions, axis=1).reshape(xx.shape)
    
    plt.contourf(xx, yy, predictions, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.RdYlBu)
    plt.show()

def plot_training_curves(epochs, train_accuracies, train_losses, val_accuracies=None, val_losses=None):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy', marker='o')
    if val_accuracies:
        plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o')
    if val_losses:
        plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()
