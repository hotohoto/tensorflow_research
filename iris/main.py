from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib.request

import numpy as np
import tensorflow as tf

# Data sets
DATA_DIR = os.path.dirname(os.path.realpath(__file__))
IRIS_TRAINING = DATA_DIR + "/iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = DATA_DIR + "/iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"


def main():
    # If the training and test sets aren't stored locally, download them.
    if not os.path.exists(IRIS_TRAINING):
        with urllib.request.urlopen(IRIS_TRAINING_URL) as url:
            raw = url.read()
            with open(IRIS_TRAINING, "wb") as f:
                f.write(raw)

    if not os.path.exists(IRIS_TEST):
        with urllib.request.urlopen(IRIS_TEST_URL) as url:
            raw = url.read()
            with open(IRIS_TEST, "wb") as f:
                f.write(raw)

    # Load datasets.
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING,
        target_dtype=np.int,
        features_dtype=np.float32)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST,
        target_dtype=np.int,
        features_dtype=np.float32)

    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    print("create classifier")
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 20, 10],
        n_classes=3)
    # Define the training inputs
    def get_train_inputs():
        x = tf.constant(training_set.data)
        y = tf.constant(training_set.target)
        return x, y

    # Fit model.
    print("fit()")
    classifier.fit(input_fn=get_train_inputs, steps=200)

    # Define the test inputs
    def get_test_inputs():
        x = tf.constant(test_set.data)
        y = tf.constant(test_set.target)
        return x, y

    # Evaluate accuracy.
    print("evaluate()")
    accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

    # Classify two new flower samples.
    def new_samples():
        return np.array(
            [[6.4, 3.2, 4.5, 1.5],
             [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)

    predictions = list(classifier.predict(input_fn=new_samples))

    print("New Samples, Class Predictions: {}\n".format(predictions))


if __name__ == "__main__":
    main()
