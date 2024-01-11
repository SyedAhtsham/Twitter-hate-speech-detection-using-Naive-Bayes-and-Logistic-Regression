import random
import sys


from sklearn.metrics import f1_score

predictions = []
def predictor(actual_class=None):
    if actual_class is not None:
        predictions.append(actual_class)
        return actual_class
    else:
        labels = ["offensive", "nonoffensive"]
        prediction = random.choice(labels)
        predictions.append(prediction)
        return prediction



def accuracy(classifier, data):
    """Computes the accuracy of a classifier on reference data.

    Args:
        classifier: A classifier.
        data: Reference data.

    Returns:
        The accuracy of the classifier on the test data, a float.
    """
    ##################### STUDENT SOLUTION #########################
    # YOUR CODE HERE

    no_correct_predictions = 0
    for sample in data:
        prediction = classifier.predict(sample[0])
        predictions.append(prediction)
        if prediction == sample[1]:
            no_correct_predictions += 1
    accuracy = no_correct_predictions/len(data)

    return accuracy
    ################################################################



def f_1(classifier, data):
    """Computes the F_1-score of a classifier on reference data.

    Args:
        classifier: A classifier.
        data: Reference data.

    Returns:
        The F_1-score of the classifier on the test data, a float.
    """
    ##################### STUDENT SOLUTION #########################
    # YOUR CODE HERE
    tp = fp = fn = 0
    predicted_labels = []
    actual_labels = []
    for sample in data:
        prediction = classifier.predict(sample[0])
        actual_labels.append(sample[1])
        predicted_labels.append(prediction)
        if sample[1] == "offensive" and prediction == "offensive":
            tp += 1
        elif sample[1] == "nonoffensive" and prediction == "offensive":
            fp += 1
        elif sample[1] == "offensive" and prediction == "nonoffensive":
            fn += 1


    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    f1 = f1_score(actual_labels, predicted_labels, average="weighted")
    print("Built-in f measure:" + str(f1))

    f_1 = (2*precision*recall)/(precision+recall)

    return f1
    ################################################################
