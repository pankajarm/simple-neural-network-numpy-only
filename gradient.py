# Simple example to imlement gradient descent and train it on USA Graduate School admissions data
#  data taken from http://www.ats.ucla.edu/stat/data/binary.csv
# Original data has School rank as 1,2,3 or 4, student GRE score, student GPA score and college admit decision (0- NO, 1- Yes)

# Data cleanup is done using data_prep.py, see data_prep.py for more details

from __future__ import division
import numpy as np
from data_prep import features, targets, features_test, targets_test

# our activation function here is sigmoid

def sigmoid(x):
    """
    Calculate the sigmoid
    """
    return 1 / (1 + np.exp(-x))

# seed same level of randomness to make debugging easier for everyone
np.random.seed(42)

n_records, n_features = features.shape
last_loss = None

# Initialize the weights
weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

# Our Neural Network hyperparameters
epochs = 1000
learnrate = 0.5

for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(features.values, targets):
        # Loop through all records, x is the input, y is the target

        # TODO: Calculate the output
        output = sigmoid(np.dot(x,weights))

        # TODO: Calculate the error
        error = y - output

        # TODO: Calculate change in weights
        del_w += error * output * (1 - output) * x

        # TODO: Update weights
    weights += learnrate * del_w / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss


# Calculate accuracy on test data
tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
