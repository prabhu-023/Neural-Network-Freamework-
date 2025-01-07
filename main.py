from dataset_utils import get_dataset
from visualizations import plot_decision_boundary, plot_training_curves
from neural_network_m import Model, Accuracy_clasification, CrossEntropy, Loss_BinaryCrossEntropy, MeanSquare_error, MeanAbsolute_error, Accuracy_regression
from optimizers import Optimizer_Adam
from dynamic_model import build_model


def get_loss_function(problem_type):
    """
    Prompts the user to choose a loss function based on the problem type.

    Args:
        problem_type (str): Type of problem ('classification' or 'regression').

    Returns:
        Loss: Configured loss function class.
    """
    if problem_type == 'classification':
        print("\nChoose a loss function for classification:")
        print("1. CrossEntropy (multiclass classification)")
        print("2. BinaryCrossEntropy (binary classification)")
        choice = input("Enter your choice (1 or 2): ").strip()

        if choice == '1':
            return CrossEntropy()
        elif choice == '2':
            return Loss_BinaryCrossEntropy()
        else:
            raise ValueError("Invalid choice for classification loss function.")

    elif problem_type == 'regression':
        print("\nChoose a loss function for regression:")
        print("1. MeanSquaredError")
        print("2. MeanAbsoluteError")
        choice = input("Enter your choice (1 or 2): ").strip()

        if choice == '1':
            return MeanSquare_error()
        elif choice == '2':
            return MeanAbsolute_error()
        else:
            raise ValueError("Invalid choice for regression loss function.")

    else:
        raise ValueError("Unsupported problem type. Use 'classification' or 'regression'.")


def main():
    # Load dataset
    dataset_name = input("Choose a dataset (spiral, sine, csv): ").strip().lower()
    X, y = get_dataset(dataset_name)
    print(X[:5], "\n")
    print(len(set(y)))
    output_size = 0

    # Determine problem type
    if len(y.shape) == 1:
        if len(set(y)) >= 2:
            problem_type = 'classification'  # Binary classification
            output_size = len(set(y))
        else:
            problem_type = 'regression'  # Regression
            output_size = len(set(y))
    elif len(y.shape) > 1:
        problem_type = 'classification'  # Multiclass classification
        output_size = y.shape[-1]
    else:
        raise ValueError("Unsupported dataset format.")

    print(f"Problem type detected: {problem_type}")

    # Prompt for loss function
    loss_function = get_loss_function(problem_type)

    # Define layers configuration for the hidden layers
    layers_config = [
        {'type': 'dense', 'neurons': 128, 'activation': 'relu', 'dropout': 0.1},
        {'type': 'dense', 'neurons': 512, 'activation': 'relu', 'dropout': 0.2},
        {'type': 'dense', 'neurons': 32, 'activation': 'relu'}
    ]

    # Build and train the model
    input_size = X.shape[1]
    model = build_model(
        input_size, output_size, layers_config,
        weight_initializer=0.1, weight_regularizer_l1=1e-4, weight_regularizer_l2=1e-3
    )

    # Set optimizer and accuracy metric
    model.set(
        loss=loss_function,
        optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5, beta1=0.9, beta2=0.999, epsilon=1e-7),
        accuracy=Accuracy_clasification() if problem_type == 'classification' else Accuracy_regression()
    )

    model.finalize()

    # Train the model
    metrics = model.train(X, y, epochs=10, printevery=100, batch_size=32, validation_data=(X, y))

    # Extract metrics for plotting
    train_accuracies, train_losses, val_accuracies, val_losses = metrics

    # Plot training and validation curves
    print("\n--- Plotting Training Curves ---")
    plot_training_curves(
        epochs=10,
        train_accuracies=train_accuracies,
        train_losses=train_losses,
        val_accuracies=val_accuracies,
        val_losses=val_losses
    )

    # Visualize decision boundary (for spiral dataset)
    if dataset_name == 'spiral' and problem_type == 'classification':
        plot_decision_boundary(model, X, y)


if __name__ == "__main__":
    main()
