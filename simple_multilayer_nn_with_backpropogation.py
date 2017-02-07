# Here we will implement a forward pass through a 4x3x2 network, with sigmoid activation functions for both layers.
# we will be using fake data made from numpy random.randn() function

import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))

# we will be using fake data
x = np.array([0.5, 0.1, -0.2])

# Here is our simple multilayer Network size
target = 0.6
learnrate = 0.5


# Now the weight from input layer to hidden
weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

# weight from hidden to output layer
weights_hidden_output = np.array([0.1, -0.3])

## Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
output = sigmoid(output_layer_in)

## Backwards pass
## TODO: Calculate error
error = target - output

# TODO: Calculate error gradient for output layer
del_err_output =  error * output * (1 - output)

# TODO: Calculate error gradient for hidden layer
del_err_hidden = weights_hidden_output * del_err_output * hidden_layer_output * ( 1 - hidden_layer_output)

# TODO: Calculate change in weights for hidden layer to output layer
delta_w_h_o = learnrate * del_err_output * hidden_layer_output

# TODO: Calculate change in weights for input layer to hidden layer
delta_w_i_o = learnrate * del_err_hidden * x[:,None]

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_o)
