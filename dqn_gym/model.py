import tensorflow as tf
import numpy as np
import random
from collections import deque

class DQN:
    LEARNING_RATE = 0.001
    REPLAY_MEMORY = 50000
    BATCH_SIZE = 64
    GAMMA = 0.9
    STATE_FRAME_LEN = 4

    def __init__(self, session, size, n_action):
        self.session = session
        self.n_action = n_action
        self.size = size
        self.memory = deque()
        self.state = None

        self.input_X = tf.placeholder(tf.float32, [None, size, self.STATE_FRAME_LEN])
        self.input_A = tf.placeholder(tf.int64, [None])
        self.input_Y = tf.placeholder(tf.float32, [None])

        self.Q = self._build_network('main')
        self.cost, self.train_op = self._build_op()

        self.target_Q = self._build_network('target')

    def _build_network(self, name):
        with tf.variable_scope(name):
            model = tf.contrib.layers.flatten(self.input_X)
            model = tf.layers.dense(model, 16, activation=tf.nn.relu)
            Q = tf.layers.dense(model, self.n_action, activation=None)

        return Q

    def _build_op(self):
        # Perform a gradient descent step on (y_j-Q(ð_j,a_j;θ))^2
        one_hot = tf.one_hot(self.input_A, self.n_action, 1.0, 0.0)
        Q_value = tf.reduce_sum(tf.multiply(self.Q, one_hot), axis=1)
        cost = tf.reduce_mean(tf.square(self.input_Y - Q_value))
        train_op = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE).minimize(cost)

        return cost, train_op

    def update_target_network(self):
        copy_op = []

        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign(main_var.value()))

        self.session.run(copy_op)

    def get_action(self):
        Q_value = self.session.run(self.Q, feed_dict={self.input_X: [self.state]})
        action = np.argmax(Q_value[0])
        return action

    def init_state(self, state):
        state = [state for _ in range(self.STATE_FRAME_LEN)]
        self.state = np.stack(state, axis=-1)

    def remember(self, state, action, reward, done):
        next_state = np.reshape(state, (self.size, 1))
        next_state = np.append(self.state[:, 1:], next_state, axis=1)

        self.memory.append((self.state, next_state, action, reward, done))

        if len(self.memory) > self.REPLAY_MEMORY:
            self.memory.popleft()

        self.state = next_state

    def _sample_memory(self):
        sample_memory = random.sample(self.memory, self.BATCH_SIZE)

        state = [memory[0] for memory in sample_memory]
        next_state = [memory[1] for memory in sample_memory]
        action = [memory[2] for memory in sample_memory]
        reward = [memory[3] for memory in sample_memory]
        done = [memory[4] for memory in sample_memory]

        return state, next_state, action, reward, done

    def train(self):
        state, next_state, action, reward, done = self._sample_memory()

        target_Q_value = self.session.run(self.target_Q, feed_dict={self.input_X: next_state})

        Y = []
        for i in range(self.BATCH_SIZE):
            if done[i]:
                Y.append(reward[i])
            else:
                Y.append(reward[i] + self.GAMMA * np.max(target_Q_value[i]))

        self.session.run(self.train_op,
                         feed_dict={
                             self.input_X: state,
                             self.input_A: action,
                             self.input_Y: Y
                         })
