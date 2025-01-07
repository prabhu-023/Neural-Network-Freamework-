# Neural Network from Scratch

This project demonstrates the implementation of a neural network from scratch using **NumPy** and custom modules for layers, activation functions, loss functions, optimizers, and more. The project supports both binary and multiclass classification tasks as well as regression problems. Synthetic datasets like `spiral_data` and `sine_data` are used for demonstration purposes.

---

## Features
- **Custom Neural Network Implementation**:
  - Fully customizable layers with support for weight regularization and dropout.
  - Flexible activation functions: ReLU, Softmax, and more.
  - Loss functions: CrossEntropy, Binary CrossEntropy, MeanSquaredError, and MeanAbsoluteError.
  - Support for binary, multiclass classification, and regression tasks.

- **Dynamic Model Building**:
  - Layers can be manually added to the model.
  - Flexible configuration for input size, neurons, and activation.

- **Dataset Support**:
  - Synthetic datasets like `spiral_data` and `sine_data`.
  - CSV datasets can also be loaded with minimal changes.

- **Visualization**:
  - Training and validation curves for accuracy and loss.
  - Decision boundary visualization for 2D datasets.

---

## Installation

### Prerequisites
- Python 3.8+
- Required Python libraries:
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `nnfs`

### Install Dependencies
Run the following command to install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Project Structure

```
project/
│
├── main.py                   # Main script to train and evaluate the model
├── neural_network_m.py       # Implementation of layers, loss, activations, accuracy
├── optimizers.py             # Implementation of optimizers (e.g., Adam)
├── visualizations.py         # Utility functions for plotting
├── README.md                 # Project documentation
└── requirements.txt          # List of dependencies
```

---

## Usage

### Running the Main Script
To run the project, execute the following command:
```bash
python main.py
```

### Workflow
1. **Dataset Selection**: Choose between `spiral_data`, `sine_data`, or a custom CSV dataset.
2. **Layer Configuration**: The model layers can be manually added in `main.py`.
3. **Training**:
   - Specify the number of epochs, batch size, and whether to validate.
4. **Visualization**:
   - Training and validation curves are plotted automatically.
   - Decision boundaries are visualized for 2D datasets.

---

## Example Run

1. Select the dataset:
   ```
   Choose a dataset (spiral or sine): spiral
   Enter the number of classes (2 or 3): 3
   Do you want to validate the model? (Y/N): Y
   ```
2. Watch the training process and metrics:
   ```
   Epoch: 1, Loss: 1.25, Accuracy: 45.3%
   Epoch: 10, Loss: 0.55, Accuracy: 85.2%
   ```
3. View the decision boundary:
   ![Decision Boundary Example](decision_boundary.png)

---

## Key Modules

### `neural_network_m.py`
Contains the implementation of core neural network components:
- Layers: Dense, Dropout.
- Activation functions: ReLU, Softmax.
- Loss functions: CrossEntropy, BinaryCrossEntropy, MeanSquaredError, MeanAbsoluteError.
- Accuracy metrics: Classification and Regression.

### `optimizers.py`
Contains implementations of:
- OptimizerAdam

### `visualizations.py`
Utility functions to:
- Plot decision boundaries for 2D datasets.
- Plot training and validation metrics.

---

## Customization
- **Dataset**: Add support for custom datasets in `main.py`.
- **Layers**: Add new types of layers or activation functions in `neural_network_m.py`.
- **Loss Functions**: Extend support for other loss functions.

---

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

---

## License
This project is licensed under the MIT License.

---

## Acknowledgements
- [nnfs.io](https://nnfs.io): For the datasets and inspiration to build neural networks from scratch.
- The Python community for providing great libraries like NumPy and Matplotlib.

