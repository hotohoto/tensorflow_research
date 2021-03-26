from functools import reduce
import os
import random
import time

import gym
import tensorflow as tf
import numpy as np

from model import DQN

GLOBAL_SEED = 11199
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
MAX_EPISODE = 40
TARGET_UPDATE_INTERVAL = 5
TRAIN_INTERVAL_FRAMES = 64

env = gym.make('CartPole-v1')
env.seed(GLOBAL_SEED)
action_size = env.action_space.n
observation_size = env.observation_space.shape[0]

def train():
    with tf.Session() as sess:
        tf.set_random_seed(GLOBAL_SEED)
        brain = DQN(sess, observation_size, action_size)
        rewards = tf.placeholder(tf.float32, [None])
        tf.summary.scalar('avg.reward/ep.', tf.reduce_mean(rewards))
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('logs', sess.graph)
        summary_merged = tf.summary.merge_all()
        brain.update_target_network()
        time_step = 0
        total_reward_list = []

        for episode in range(MAX_EPISODE):
            done = False
            total_reward = 0
            epsilon = 1. / ((episode / 10) + 1)

            observation = env.reset()
            brain.init_state(observation)

            while not done:
                if np.random.rand() < epsilon:
                    action = random.randrange(action_size)
                else:
                    action = brain.get_action()

                observation, reward, done, _ = env.step(action)
                # print(observation, reward, done)
                total_reward += reward
                brain.remember(observation, action, reward, done)

                if time_step > 0:
                    if time_step % TRAIN_INTERVAL_FRAMES == 0:
                        _, loss = brain.train()
                    if time_step % TARGET_UPDATE_INTERVAL == 0:
                        brain.update_target_network()

                time_step += 1

            print('episode: %d total_reward: %d' % (episode, total_reward))

            total_reward_list.append(total_reward)

            if episode % 10 == 0:
                summary = sess.run(summary_merged, feed_dict={rewards: total_reward_list})
                writer.add_summary(summary, time_step)
                total_reward_list = []

            if episode % 100 == 0:
                saver.save(sess, 'model/dqn.ckpt', global_step=time_step)


def replay():
    sess = tf.Session()
    brain = DQN(sess, observation_size, action_size)

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('model')
    saver.restore(sess, ckpt.model_checkpoint_path)

    for episode in range(MAX_EPISODE):
        done = False
        total_reward = 0
        observation = env.reset()
        brain.init_state(observation)

        while not done:
            action = brain.get_action()
            observation, reward, done, _ = env.step(action)
            total_reward += reward
            brain.remember(observation, action, reward, done)
            time.sleep(0.3)

        print('episode: %d total_reward: %d' % (episode, total_reward))

if __name__ == '__main__':
    _train = os.environ.get('TRAIN')

    if _train == None or _train.lower() == 'y':
        train()
    else:
        replay()
