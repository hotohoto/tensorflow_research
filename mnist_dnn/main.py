# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners

Originated from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir)

    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("pixels", dimension=784)]

    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        mnist.test.images,
        mnist.test.labels,
        every_n_steps=1,
        early_stopping_metric="loss",
        early_stopping_metric_minimize=True,
        early_stopping_rounds=200)

    # Build 3 layer DNN
    print("create classifier")
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[784, 784],
        n_classes=10
    )

    # Define the training inputs
    def get_train_inputs():
        batch_xs, batch_ys = mnist.train.next_batch(6000)
        x = tf.constant(batch_xs)
        y = tf.cast(tf.constant(batch_ys), tf.int64)
        return {"pixels": x}, y

    # Fit model.
    print("fit()")
    tf.logging.set_verbosity(tf.logging.INFO)
    classifier.fit(input_fn=get_train_inputs, steps=200)

    # Define the test inputs
    def get_test_inputs():
        x = tf.constant(mnist.test.images)
        y = tf.cast(tf.constant(mnist.test.labels), tf.int64)
        return {"pixels": x}, y

    # Evaluate accuracy.
    print("evaluate()")
    evaluation = classifier.evaluate(input_fn=get_test_inputs, steps=1)

    print("\nEvaluation:", evaluation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
