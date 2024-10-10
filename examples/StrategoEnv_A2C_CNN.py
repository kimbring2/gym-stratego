import os
import random
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import cv2
import threading
from threading import Thread, Lock
import time
import tensorflow_probability as tfp
from typing import Any, List, Sequence, Tuple

from gym_stratego.envs import StrategoEnv
import gym_stratego
from gym_stratego.envs.constants import *
import utils

env = gym.make("stratego-v0")

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


tfd = tfp.distributions


class ActorCritic(tf.keras.Model):
    def __init__(self, action_space):
        super(ActorCritic, self).__init__()

        self.conv_1 = Conv2D(64, 2, 4, padding="valid", activation="relu")
        self.conv_2 = Conv2D(512, 3, 1, padding="valid", activation="relu")

        self.dense_1 = Dense(512, activation='relu')

        self.policy = Dense(action_space)
        self.value = Dense(1)

    def call(self, state, action_mask):
        # state.shape:  (1, 10, 10, 25)
        # action_mask.shape:  (1, 1856)

        conv_1 = self.conv_1(state)
        conv_2 = self.conv_2(conv_1)
        conv_2_flattened = Flatten()(conv_2)

        action_mask = Flatten()(action_mask)
        dense_1 = self.dense_1(action_mask)

        dense = tf.concat([conv_2_flattened, dense_1], axis=1)

        action_logit = self.policy(dense)
        value = self.value(dense)

        return action_logit, value


mse_loss = tf.keras.losses.MeanSquaredError()


class A3CAgent:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = env = gym.make("stratego-v0")
        self.action_size = env.action_space.n
        self.EPISODES, self.episode, self.max_average = 2000000, 0, -21.0 # specific for pong
        self.lr = 0.0001

        self.ROWS = 64
        self.COLS = 64
        self.REM_STEP = 4

        # Instantiate plot memory
        self.scores, self.episodes, self.average = [], [], []
        self.state_size = (self.ROWS, self.COLS, self.REM_STEP)

        # Create Actor-Critic network model
        self.model = ActorCritic(action_space=self.action_size)
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.writer = tf.summary.create_file_writer("tensorboard")

    def act(self, state):
        # Use the network to predict the next action to take, using the model
        prediction = self.model(state, training=False)
        action = tf.random.categorical(prediction[0], 1).numpy()

        return action[0][0]

    def discount_rewards(self, rewards):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.95    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(rewards)
        for i in reversed(range(0, len(rewards))):
            if rewards[i] != 0: # reset the sum, since this was a game boundary (pong specific!)
                running_add = 0

            running_add = running_add * gamma + rewards[i]
            discounted_r[i] = running_add

        if np.std(discounted_r) != 0.0:
            discounted_r -= np.mean(discounted_r) # normalizing the result
            discounted_r /= np.std(discounted_r) # divide by standard deviation

        return discounted_r

    def replay(self, states, actions, rewards, action_masks):
        states = np.vstack(states)
        action_masks = np.vstack(action_masks)

        discounted_r = self.discount_rewards(rewards)
        with tf.GradientTape() as tape:
            prediction = self.model(states, action_masks, training=True)
            action_logits = prediction[0]
            values = prediction[1]

            advantages = discounted_r - np.stack(values)[:, 0]

            action_logits *= action_masks
            dist = tfd.Categorical(logits=action_logits)
            action_log_prob = dist.prob(actions)
            action_log_prob = tf.math.log(action_log_prob)

            actor_loss = -tf.math.reduce_mean(action_log_prob * advantages)

            critic_loss = mse_loss(values, np.vstack(discounted_r))
            critic_loss = tf.cast(critic_loss, 'float32')
            
            total_loss = actor_loss + 0.5 * critic_loss
            print("total_loss: ", total_loss)

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        return self.average[-1]

    def one_hot(self, a, num_classes):
        return np.squeeze(np.eye(num_classes)[a])

    def battle_field_one_hot(self, battle_field):
        #print(battle_field)

        state = np.zeros([10, 10, 26], dtype=np.float32)
        for i in range(0, 10):
            for j in range(0, 10):
                state[i][j][int(battle_field[i][j])] = 1.0

        state = np.expand_dims(state, 0)

        return state

    def train(self):
        while self.episode < self.EPISODES:
            obs = self.env.reset()

            # Reset episode
            score, done, SAVING = 0, False, ''

            states, actions, rewards, action_masks = [], [], [], []
            step = 0 
            while True:
                #self.env.update_screen()

                if self.env.turn == 'Blue':
                    (oldlocation, move) = self.env.brains[self.env.turn].findMove()
                    result = self.env.move_unit(oldlocation[0], oldlocation[1])
                    result = self.env.move_unit(move[0], move[1])
                    next_obs, reward, _, info = self.env.small_observation()
                    done = self.env.done
                    reward = self.env.reward[0]
                    #print("done 1: ", done)

                    if done:
                        rewards[-1] = -1.0
                        score += -1.0
                        break
                else:
                    battle_field = obs['battle_field']
                    possible_actions = obs['possible_actions']
                    ego_offboard = obs['ego_offboard']
                    oppo_offboard = obs['oppo_offboard']

                    action_mask = np.zeros(self.action_size, dtype=np.float64)
                    for possible_action in possible_actions:
                        action_mask[possible_action] = 1.0

                    battle_field = np.reshape(battle_field, (10, 10))
                    state = self.battle_field_one_hot(battle_field)

                    prediction = self.model(state, np.expand_dims(action_mask, 0), training=False)
                    fn_pi = prediction[0]

                    fn_pi = tf.nn.softmax(fn_pi)
                    fn_pi *= action_mask

                    fn_probs = fn_pi / tf.reduce_sum(fn_pi, axis=1, keepdims=True)
                    fn_dist = tfd.Categorical(probs=fn_probs)
                    action = fn_dist.sample()[0]

                    next_obs, _, _, info = self.env.step(action)
                    done = self.env.done
                    reward = self.env.reward[0]
                    reward = float(reward)

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    action_masks.append(action_mask)

                    score += reward
                    if done:
                        break

                obs = next_obs
                step += 1
                #print("")

            self.replay(states, actions, rewards, action_masks)
            states, actions, rewards, action_masks = [], [], [], []

            average = self.PlotModel(score, self.episode)
            
            #self.model.save_weights("multi_pong_server_{}.h5".format(self.episode))

            # saving best models
            if average >= self.max_average:
                self.max_average = average
                SAVING = "SAVING"
            else:
                SAVING = ""

            print("episode: {}/{}, score: {}, average: {} {}".format(self.episode, self.EPISODES, score, average, SAVING))
            #with self.writer.as_default():
            #    tf.summary.scalar("server, average_reward", average, step=self.episode)
            #    self.writer.flush()
            
            if self.episode < self.EPISODES:
                self.episode += 1


        env.close()

    def test(self, Actor_name, Critic_name):
        self.load(Actor_name, Critic_name)
        for e in range(100):
            state = self.reset(self.env)
            done = False
            score = 0
            while not done:
                self.env.render()
                action = np.argmax(self.Actor.predict(state))
                state, reward, done, _ = self.step(action, self.env, state)
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, score))
                    break

        self.env.close()


if __name__ == "__main__":
    env_name = 'PongDeterministic-v4'
    #env_name = 'Pong-v0'
    agent = A3CAgent(env_name)
    agent.train() # use as A3C