import os
import random
import gym
import numpy as np
import cv2
import time
from typing import Any, List, Sequence, Tuple

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import tensorflow_probability as tfp

from gym_stratego.envs import StrategoEnv
import gym_stratego

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tfd = tfp.distributions

writer = tf.summary.create_file_writer("tensorboard/bootstrapping")


class OurModel(tf.keras.Model):
    def __init__(self, input_shape, action_space):
        super(OurModel, self).__init__()
        
        self.common_1 = Dense(512, activation="relu")
        self.common_2 = Dense(512, activation="relu")
        self.common_3 = Dense(512, activation="relu")
        
        self.dense_1 = Dense(action_space)
        self.dense_2 = Dense(1)
        
    def call(self, state):
        state_flatten = Flatten()(state)

        common_1 = self.common_1(state_flatten)
        common_2 = self.common_2(common_1)
        common_3 = self.common_3(common_2)
        
        action_logit = self.dense_1(common_3)
        value = self.dense_2(common_3)
        
        return action_logit, value

mse_loss = tf.keras.losses.MeanSquaredError()

class A3CAgent:
    def __init__(self, env_name):
        self.env_name = env_name       
        self.env = gym.make(env_name)

        self.action_size = self.env.action_space.n
        self.EPISODES, self.episode, self.max_average = 2000000, 0, -21.0 # specific for pong
        self.lr = 0.0001

        self.ROWS = 80
        self.COLS = 80
        self.REM_STEP = 4

        self.scores, self.episodes, self.average = [], [], []

        self.Save_Path = 'Models'
        self.state_size = (self.REM_STEP, self.ROWS, self.COLS)
        
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = '{}_BOOT_{}'.format(self.env_name, self.lr)
        self.model_name = os.path.join(self.Save_Path, self.path)

        self.ActorCritic = OurModel(input_shape=self.state_size, action_space=self.action_size)
        
        self.learning_rate = 0.0001
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

        self.eps = np.finfo(np.float32).eps.item()

    @tf.function
    def act(self, state):
        prediction = self.ActorCritic(state, training=False)
        action = tf.random.categorical(prediction[0], 1)
        return action[0][0]

    def get_expected_return(self, rewards, dones, gamma: float=0.99, standardize: bool=True):
        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)
    
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        dones = tf.cast(dones[::-1], dtype=tf.bool)
        
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            done = dones[i]
            if tf.cast(done, tf.bool):
                discounted_sum = tf.constant(0.0)
            
            discounted_sum = reward + gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        
        returns = returns.stack()[::-1]
        
        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + self.eps))
    
        return returns

    @tf.function
    def replay(self, states, actions, rewards, dones):
        discounted_r = self.get_expected_return(rewards, dones)
        with tf.GradientTape() as tape:
            #tf.print("states.shape: ", states.shape)
            prediction = self.ActorCritic(states, training=True)
            action_logits = prediction[0]
            values = prediction[1]
            
            #tf.print("discounted_r.shape: ", discounted_r.shape)
            #tf.print("tf.stack(values)[:, 0].shape: ", tf.stack(values)[:, 0].shape)
            advantages = discounted_r - tf.stack(values)[:, 0]
            
            action_probs = tf.nn.softmax(action_logits)
            dist = tfd.Categorical(probs=action_probs)
            action_log_prob = dist.log_prob(actions)
    
            actor_loss = -tf.math.reduce_mean(action_log_prob * advantages)
            
            #tf.print("values.shape: ", values.shape)
            #tf.print("")
            critic_loss = mse_loss(values, tf.stack(discounted_r))
            critic_loss = tf.cast(critic_loss, 'float32')
    
            entropy_loss = dist.entropy()
            entropy_loss = tf.math.reduce_mean(entropy_loss)
    
            entropy_loss = 0.001 * -entropy_loss
            actor_loss += entropy_loss

            total_loss = actor_loss + critic_loss
    
        grads = tape.gradient(total_loss, self.ActorCritic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.ActorCritic.trainable_variables))
        
    def load(self, model_name):
        self.ActorCritic = load_model(model_name, compile=False)

    def save(self):
        self.ActorCritic.save(self.model_name)

    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        return self.average[-1]
    
    def train(self):
        score, done, SAVING = 0, False, ''
        state = self.env.reset()

        battle_field = state['battle_field']
        battle_field = np.reshape(battle_field, (10, 10))
        possible_actions = state['possible_actions']

        while self.episode < self.EPISODES:
            states, actions, rewards, dones = [], [], [], []
            for step in range(0, 512):
                start = time.time()

                print("battle_field.shape: ", battle_field.shape)
                print("possible_actions: ", possible_actions)

                action = self.act(np.expand_dims(battle_field, 0))
                next_state, reward_both, done, _ = self.env.step(action.numpy())

                reward = 0
                if reward_both[0] == 1:
                    reward = 1
                elif reward_both[1] == 1:
                    reward = -1
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)

                score += reward
                state = next_state

                if done:
                    average = self.PlotModel(score, self.episode)
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, score, average, SAVING))

                    score, done, SAVING = 0, False, ''
                    state = self.reset()

                end = time.time()
                print("elapsed time: ", end - start)
                print("")
                    
            self.replay(np.array(states), np.array(actions), np.array(rewards), np.array(dones))
            states, actions, rewards, dones = [], [], [], []
                
            if self.episode < self.EPISODES:
                self.episode += 1

        self.env.close()            

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
    env_name = 'stratego-v0'
    agent = A3CAgent(env_name)
    agent.train() # use as A3C