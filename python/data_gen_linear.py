'''
This module is used to generate the training and test data for linear regression study.
'''
import sys
import numpy as np
import pandas as pd

def generation(w, b, num_of_examples):
    num_of_features = len(w)
    # Generate data
    features = np.random.normal(0, 1, (num_of_examples, num_of_features))
    labels = np.matmul(features, w.T)
    # Add noise
    labels += np.random.normal(0, 0.01, size=len(labels))
    return np.column_stack((features, labels))

def main():
    '''
    Main entry of the module as the main program
    '''
    # True parameters
    w = np.array([2, -3.4])
    b = np.array([4.2])
    # Data generation
    data = generation(w, b, 100000)
    # Store to file
    data_frame = pd.DataFrame(data, columns=None, index=None)
    data_frame.to_csv('../Data/linear_reg.dat')

if __name__ == '__main__':
    main()
    sys.exit(0)
