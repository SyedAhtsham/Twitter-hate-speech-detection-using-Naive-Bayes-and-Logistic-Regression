import copy
import math
import sys


from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


class NaiveBayes(object):

    ######################### STUDENT SOLUTION #########################
    # YOUR CODE HERE
    def __init__(self):
        """Initialises a new classifier."""
        self.prior_probs = {}
        self.cond_probs = {}
        self.classes = []
        self.words = []


    ####################################################################


    def predict(self, x):
        """Predicts the class for a document.

        Args:
            x: A document, represented as a list of words.

        Returns:
            The predicted class, represented as a string.
        """
        ################## STUDENT SOLUTION ########################
        # YOUR CODE HERE
        probs = {}




        # print(x)

        # print(self.cond_probs)

        # print(words)
        for c in self.classes:
            sum = 0
            for token in x:
                if token in self.words:
                    sum += self.cond_probs[c][token]
            probs[c] = sum

        # print(prob)
        return max(probs, key=probs.get)
        ############################################################


    def count_words(self, vocabulary, mega_text):
        words_counts = {}
        for type in vocabulary:
            words_counts[type] = 0
            for token in mega_text:
                if type == token:
                    words_counts[type] += 1
                    mega_text.remove(token)
        return words_counts

    @classmethod
    def train(cls, data, k=1):
        """Train a new classifier on training data using maximum
        likelihood estimation and additive smoothing.

        Args:
            cls: The Python class representing the classifier.
            data: Training data.
            k: The smoothing constant.

        Returns:
            A trained classifier, an instance of `cls`.
        """



        ##################### STUDENT SOLUTION #####################
        # YOUR CODE HERE
        instance = cls()

        vocabulary = set()
        all_words = []
        C = set()
        for sample in data:
            C.add(sample[1])
            all_words.extend(sample[0])
            vocabulary.update(sample[0])

        tweets = {}
        temp_data = copy.deepcopy(data)
        instance.classes = sorted(C)

        texts = {}
        # for c in instance.classes:

        for c in instance.classes:
            tweets[c] = []
            texts[c] = []


        for sample in temp_data:
            tweets[sample[1]].append(sample[0])
            texts[sample[1]].extend(sample[0])
            # tweets[c].update(sample[0]) if we consider it as a set                temp_data.remove(sample)

        for c in instance.classes:
            instance.prior_probs[c] = math.log(len(tweets[c])/len(data))


        word_frequency = {}
        mega_text = copy.deepcopy(texts)


        for c in instance.classes:
            word_frequency[c] = instance.count_words(vocabulary, texts[c])



        # print(word_frequency)
        # Print the word counts for each class
        for class_name, word_counts in word_frequency.items():
            # print(f'Class: {class_name}')
            word_prob = {}
            for word, count in word_counts.items():
                # print(f'  {word}: {count}')
                word_prob[word] = math.log((count+k)/(len(texts[class_name])+(k*len(vocabulary))))
            instance.cond_probs[class_name] = word_prob

            # print()

        # Likelihoods from training dataset
        instance.words = [word for word_counts in instance.cond_probs.values() for word in word_counts.keys()]


        return instance
        ############################################################



def features1(data, k=1):
    """
    Your feature of choice for Naive Bayes classifier.

    Args:
        data: Training data.
        k: The smoothing constant.

    Returns:
        Parameters for Naive Bayes classifier, which can
        then be used to initialize `NaiveBayes()` class
    """
    ###################### STUDENT SOLUTION ##########################
    # YOUR CODE HERE


    # Read the set of stop words from the text file
    stop_words = set()
    with open("stop_words.txt", "r") as file:
        for line in file:
            stop_words.add(line.strip())


    for sample in data:
        for word in stop_words:
            if word in sample[0]:
                sample[0].remove(word)

    return data

    ##################################################################


def features2(data, k=1):
    """
    Your feature of choice for Naive Bayes classifier.

    Args:
        data: Training data.
        k: The smoothing constant.

    Returns:
        Parameters for Naive Bayes classifier, which can
        then be used to initialize `NaiveBayes()` class
    """
    ###################### STUDENT SOLUTION ##########################
    # YOUR CODE HERE

    stemmed_data = []
    ps = PorterStemmer()
    for sample in data:
        stemmed_data.append(([ps.stem(word) for word in sample[0]], sample[1]))


    return stemmed_data
    ##################################################################

