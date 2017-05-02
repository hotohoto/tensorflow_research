# Copyright 2017 Google, Inc. All Rights Reserved.
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
import os
import tensorflow as tf
import urllib.request
import ssl

LOGDIR = '/tmp/mnist_tutorial/'
GIST_URL = 'https://gist.githubusercontent.com/dandelionmane/4f02ab8f1451e276fea1f165a20336f1/raw/dfb8ee95b010480d56a73f324aca480b3820c180/'

### MNIST EMBEDDINGS ###
mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + 'data', one_hot=True)

### Get a sprite and labels file for the embedding projector ###
context = ssl._create_unverified_context()

with urllib.request.urlopen(GIST_URL + 'labels_1024.tsv', context=context) as url:
    raw = url.read()
    with open(LOGDIR + 'labels_1024.tsv', "wb") as f:
        f.write(raw)
with urllib.request.urlopen(GIST_URL + 'sprite_1024.png', context=context) as url:
    raw = url.read()
    with open(LOGDIR + 'sprite_1024.png', "wb") as f:
        f.write(raw)

def conv_layer(input, size_in, size_out):
    w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
    return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def fc_layer(input, size_in, size_out):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    act = tf.nn.relu(tf.matmul(input, w) + b)
    return act

def mnist_model(learning_rate, use_two_conv, use_two_fc, hparam):
    tf.reset_default_graph()

    # Setup placeholders, and reshape the data
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

    if use_two_conv:
        conv1 = conv_layer(x_image, 1, 32)
        conv_out = conv_layer(conv1, 32, 64)
    else:
        conv1 = conv_layer(x_image, 1, 64)
        conv_out = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])

    if use_two_fc:
        fc1 = fc_layer(flattened, 7 * 7 * 64, 1024)
        logits = fc_layer(fc1, 1024, 10)
    else:
        logits = fc_layer(flattened, 7 * 7 * 64, 10)

    xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name="xent")
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        batch = mnist.train.next_batch(100)
        _, _xent = sess.run([train_step, xent], feed_dict={x: batch[0], y: batch[1]})
        if i % 5 == 0:
            [train_accuracy] = sess.run([accuracy], feed_dict={x: batch[0], y: batch[1]})
            print("Step", i, train_accuracy, _xent)
        else:
            print("Step", i, "----", _xent)

def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
    conv_param = "conv=2" if use_two_conv else "conv=1"
    fc_param = "fc=2" if use_two_fc else "fc=1"
    return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)


def main():
    # You can try adding some more learning rates
    for learning_rate in [1E-4]:

        # Include "False" as a value to try different model architectures
        for use_two_fc in [True]:
            for use_two_conv in [True]:
                # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2)
                hparam = make_hparam_string(learning_rate, use_two_fc, use_two_conv)
                print('Starting run for %s' % hparam)

                # Actually run with the new settings
                mnist_model(learning_rate, use_two_fc, use_two_conv, hparam)


if __name__ == '__main__':
    main()
