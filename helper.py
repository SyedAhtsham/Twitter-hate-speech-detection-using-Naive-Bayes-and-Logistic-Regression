import matplotlib.pyplot as plt
import time

import numpy as np
from sklearn.metrics import accuracy_score

from model.naivebayes import NaiveBayes, features1, features2
from model.logreg import LogReg, featurize
from evaluation import accuracy, f_1

MODEL_DICT = {'naive-bayes': NaiveBayes, 'logreg': LogReg}

def train_smooth(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) Re-train Naive Bayes while varying smoothing parameter k,
    #         then evaluate on test_data.
    #         2) Plot a graph of the accuracy and/or f-score given
    #         different values of k and save it, don't forget to include
    #         the graph for your submission.

    ######################### STUDENT SOLUTION #########################
    model = MODEL_DICT['naive-bayes']
    values_of_k = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    # values_of_k = [1, 10, 100, 1000, 10000]

    f_1_values = []
    accuracy_values = []

    for k in values_of_k:
        nb = model.train(train_data, k)
        accuracy_values.append(round(accuracy(nb, test_data)))
        f_1_values.append(round(f_1(nb, test_data)))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(values_of_k, accuracy_values, label='Accuracy', marker='o')
    plt.plot(values_of_k, f_1_values, label='F1 Score', marker='o')

    # Add labels and legend
    plt.xlabel('Smoothing Parameter (k)')
    plt.ylabel('Metric Value')
    plt.title('Effect of Smoothing Parameter on Accuracy and F1 Score')
    plt.legend()

    # Save the plot
    # plt.savefig('plot_'+str(time.time())+".png")
    plt.savefig("plot.png")
    # Show the plot
    plt.show()



    ####################################################################



def train_feature_eng(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) Improve on the basic bag of words model by changing
    #         the feature list of your model. Implement at least two
    #         variants using feature1 and feature2
    ########################### STUDENT SOLUTION ########################

    # using features1 for training data
    processed_train_data = features1(train_data)

    # Use features1 for testing
    processed_test_data = features1(test_data)

    # Train the NaiveBayes classifier
    nb_classifier = NaiveBayes()


    # Test the classifier
    nb = nb_classifier.train(processed_train_data)
    print("Accuracy: ", round(accuracy(nb, processed_test_data), 2))
    print("F_1: ", round(f_1(nb, processed_test_data), 2))




    # Use features2 for training
    processed_train_data = features2(train_data)

    # Use features2 for testing
    processed_test_data = features2(test_data)

    # Train the NaiveBayes classifier
    nb_classifier = NaiveBayes()


    # Test the classifier
    nb = nb_classifier.train(processed_train_data)
    print("Accuracy: ", round(accuracy(nb, processed_test_data), 2))
    print("F_1: ", round(f_1(nb, processed_test_data), 2))


    #####################################################################



def train_logreg(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) First, assign each word in the training set a unique integer index
    #         with `buildw2i()` function (in model/logreg.py, not here)
    #         2) Now that we have `buildw2i`, we want to convert the data into
    #         matrix where the element of the matrix is 1 if the corresponding
    #         word appears in a document, 0 otherwise with `featurize()` function.
    #         3) Train Logistic Regression model with the feature matrix for 10
    #         iterations with default learning rate eta and L2 regularization
    #         with parameter C=0.1.
    #         4) Evaluate the model on the test set.
    ########################### STUDENT SOLUTION ########################

    # Extract features and labels for training
    X_train, Y_train = featurize(train_data, train_data=train_data)

    # Extract features and labels for testing
    X_test, Y_test = featurize(test_data, train_data=train_data)

    # Initialize and train the Logistic Regression model
    logreg_model = LogReg()
    logreg_model.train(X_train, Y_train)

    # Make predictions on the training set
    train_predictions = logreg_model.predict(X_train)

    # Make predictions on the testing set
    test_predictions = logreg_model.predict(X_test)

    # Calculate accuracy on training and testing sets
    train_accuracy = accuracy_score(np.argmax(Y_train, axis=1), train_predictions)
    test_accuracy = accuracy_score(np.argmax(Y_test, axis=1), test_predictions)

    print("Accuracy on Training Dataset: ", round(train_accuracy, 2))
    print("Accuracy on Testing Dataset: ", round(test_accuracy, 2))


    #####################################################################
