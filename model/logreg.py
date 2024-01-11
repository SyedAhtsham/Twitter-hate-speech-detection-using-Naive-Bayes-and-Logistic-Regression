import sys

import numpy as np


class LogReg:
    def __init__(self, eta=0.01, num_iter=30, batch_size=100, C=0.1):
        self.eta = eta
        self.num_iter = num_iter
        self.weights = None
        self.bias = None
        self.batch_size = batch_size
        self.C = C

    def softmax(self, inputs):
        """
        Calculate the softmax for the give inputs (array)
        :param inputs:
        :return:
        """
        # TODO: adapt for your solution

        exp_inputs = np.exp(inputs)
        sum_exp_inputs = float(sum(exp_inputs))
        softmax_result = exp_inputs / sum_exp_inputs
        return softmax_result


    def train(self, X, Y):

        #################### STUDENT SOLUTION ###################

        # weights initialization
        self.weights = np.zeros(X.shape[1])

        num_instances = X.shape[0]

        for epoch in range(self.num_iter):
            # Shuffle the data
            indices = np.random.permutation(num_instances)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            for i in range(0, num_instances, self.batch_size):
                # Select a minibatch
                X_batch = X_shuffled[i:i + self.batch_size]
                Y_batch = Y_shuffled[i:i + self.batch_size]


                probabilites = self.p(X_batch)

                # Compute the gradient of the loss with L2 regularization
                gradient = np.dot(X_batch.T, probabilites - Y_batch) + (1 / self.C) * self.weights

                # Update the weights using gradient descent
                self.weights -= self.eta * gradient

        return None
        #########################################################


    def p(self, X):
        # YOUR CODE HERE
        #     TODO:
        #         1) Fill in (log) probability prediction
        ################## STUDENT SOLUTION ########################

        #compute the log odds
        log_odds = np.dot(X, self.weights)

        # Compute the predicted probabilities using softmax
        probabilities = self.softmax(log_odds)

        return probabilities
        ############################################################


    def predict(self, X):
        # YOUR CODE HERE
        #     TODO:
        #         1) Replace next line with prediction of best class
        ####################### STUDENT SOLUTION ####################
        # Get the predicted probabilities
        probabilities = self.p(X)

        # Choose the class with the highest probability as the prediction
        prediction = np.argmax(probabilities, axis=1)

        return prediction
        #############################################################


def buildw2i(vocab):
    """
    Create indexes for 'featurize()' function.

    Args:
        vocab: vocabulary constructed from the training set.

    Returns:
        Dictionaries with word as the key and index as its value.
    """
    # YOUR CODE HERE
    #################### STUDENT SOLUTION ######################

    return {word: idx for idx, word in enumerate(vocab)}

    ############################################################


def featurize(data, train_data=None):
    """
    Convert data into X and Y where X is the input and
    Y is the label.

    Args:
        data: Training or test data.
        train_data: Reference data to build vocabulary from.

    Returns:
        Matrix X and Y.
    """
    # YOUR CODE HERE
    ##################### STUDENT SOLUTION #######################

    vocabulary = set(word for tweet, _ in train_data for word in tweet)
    words_dic = buildw2i(vocabulary)

    num_instances = len(train_data)
    num_features = len(words_dic)

    X = np.zeros((num_instances, num_features))
    Y = np.zeros((num_instances, 2))

    for i, (tweet, label) in enumerate(train_data):
        feature_vector = np.zeros(num_features)

        for word in tweet:
            if word in words_dic:
                feature_vector[words_dic[word]] = 1

        X[i] = feature_vector

        if label == "offensive":
            Y[i] = [1, 0]
        else:
            Y[i] = [0, 1]

    return X, Y

    ##############################################################

