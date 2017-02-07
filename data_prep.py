# This is supporting file for gradient.py to clean and prepare raw data from http://www.ats.ucla.edu/stat/data/binary.csv
# and get it ready for gradient.py

# Here are steps for data cleaning and data preparation 

# we actually need to transform the data first. The rank feature is categorical, the numbers don't encode any sort of relative values.
# Rank 2 is not twice as much as rank 1, rank 3 is not 1.5 more than rank 2.
# Instead, we need to use dummy variables to encode rank, splitting the data into four new columns encoded with ones or zeros.
# Rows with rank 1 have one in the rank 1 dummy column, and zeros in all other columns.
# Rows with rank 2 have one in the rank 2 dummy column, and zeros in all other columns. And so on.

# We'll also need to standardize the GRE and GPA data, which means to scale the values such they have zero mean and a standard deviation of 1.
# This is necessary because the sigmoid function squashes really small and really large inputs.
# The gradient of really small and large inputs is zero, which means that the gradient descent step will go to zero too.
# Since the GRE and GPA values are fairly large, we have to be really careful about how we initialize the weights
 # or the gradient descent steps will die off and the network won't train.
# Instead, if we standardize the data, we can initialize the weights easily and everyone is happy.

import numpy as np
import pandas as pd

admissions = pd.read_csv('binary.csv')

# Make dummy variables for rank
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
data = data.drop('rank', axis=1)

# Standarize features
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:,field] = (data[field]-mean)/std
    
# Split off random 10% of the data for testing
np.random.seed(42)
sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
data, test_data = data.ix[sample], data.drop(sample)

# Split into features and targets
features, targets = data.drop('admit', axis=1), data['admit']
features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']