
import numpy as np
import pandas as pd
import random


class Perceptron:
    """ Perceptron classifier """

    def __init__(self, n_inputs, learning_rate, iterations):
        self.n_inputs = n_inputs
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w_ = np.zeros(1)
        self.errors_ = []


    def train(self, X, y):
        """
            Fits the training data to the Perceptron
        """      
        
        #TODO: Code the Perceptron Learning Algorithm
        # (1)	Initialize weights to zero or a small random number.
        self.w_ = np.random.rand(len(X[0])+1)
        print(self.w_)
        
        for i in range(self.iterations):
            errors = 0
            for x_item, y_actual in zip(X, y):
                # activate
                predicted = self.activate(x_item)
                error = self.learning_rate*(y_actual - predicted)
                #update all the weights except the bias
                self.w_[1:] += error * x_item
                #for bias is error*1
                self.w_[0] += error

                #increase error count for the batch
                errors += int(error != 0.0)

            #add the error count of the batch to the errors variable
            self.errors_.append(errors)
            

    def net_input(self, X):
        """ Calculate the Net Input """
        # TODO: Calculate the sum of the product of each input and each weight
        result = np.dot(X, self.w_[1:]) + self.w_[0]
        return result


    def activate(self, X):
        """ STEP FUNCTION: Returns the class label after the unit step """
        # TODO: Return 1 if net_input(X) >= 0.0 or -1 otherwise
        if self.net_input(X) >= 0.0:
            return 1
        else:
            return -1
