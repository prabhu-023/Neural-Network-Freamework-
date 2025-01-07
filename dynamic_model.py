from neural_network_m import Model, LayerDense, Activation_ReLU, Soft_max, Activation_liner, sigmoid, Dropout

def build_model(input_size, output_size, layers_config, weight_initializer=0.5, weight_regularizer_l1=0.0, weight_regularizer_l2=0.0,bias_regularisationl2=0.0):
    """
    Dynamically builds a model based on user-defined layers and configurations.

    Args:
        input_size (int): Number of input features.
        output_size (int): Number of output neurons (problem dependent).
        layers_config (list of dict): Configuration for each hidden layer in the model.
            Example: [
                {'type': 'dense', 'neurons': 64, 'activation': 'relu', 'dropout': 0.2},
                {'type': 'dense', 'neurons': 32, 'activation': 'relu'}
            ]
        weight_initializer (float): Initial weight scaling factor.
        weight_regularizer_l1 (float): L1 regularization for weights.
        weight_regularizer_l2 (float): L2 regularization for weights.

    Returns:
        Model: Configured neural network model.
    """
    valid_activations = {'relu', 'softmax', 'sigmoid', 'linear'}
    model = Model()
    prev_neurons = input_size

    # Add hidden layers based on configuration
    for i, layer in enumerate(layers_config):
        # Validate the layer type
        if layer['type'] != 'dense':
            raise ValueError(f"Unsupported layer type '{layer['type']}' in layer {i+1}. Only 'dense' is supported.")

        # Validate the activation function
        if 'activation' in layer and layer['activation'] not in valid_activations:
            raise ValueError(f"Unsupported activation '{layer['activation']}' in layer {i+1}. "
                             f"Valid options: {valid_activations}.")

        # Validate dropout rate
        if 'dropout' in layer and not (0 <= layer['dropout'] <= 1):
            raise ValueError(f"Invalid dropout rate {layer['dropout']} in layer {i+1}. Must be between 0 and 1.")

        # Add Dense layer
        neurons = layer['neurons']
        model.add(LayerDense(
            prev_neurons, neurons,
            weight_initializer=weight_initializer,
            weight_regularisationl2=weight_regularizer_l2,
            bias_regularisationl2=bias_regularisationl2
        ))
        prev_neurons = neurons

        # Add activation layer
        if 'activation' in layer:
            if layer['activation'] == 'relu':
                model.add(Activation_ReLU())
            elif layer['activation'] == 'softmax':
                model.add(Soft_max())
            elif layer['activation'] == 'sigmoid':
                model.add(sigmoid())
            elif layer['activation'] == 'linear':
                model.add(Activation_liner())

        # Add dropout layer if specified
        if 'dropout' in layer:
            model.add(Dropout(rate=layer['dropout']))

    # Automatically add the output layer
    model.add(LayerDense(
        prev_neurons, output_size,
        weight_initializer=weight_initializer,
        weight_regularisationl2=weight_regularizer_l2
    ))

    # Automatically set the activation for the output layer based on output neurons
    if output_size == 1:
        # Binary classification or regression
        model.add(sigmoid())
        
    elif output_size > 1:
        # Multiclass classification
        model.add(Soft_max())
    else:
        raise ValueError("Output size must be a positive integer.")
    print(model.layers)

    return model
