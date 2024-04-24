import gym
import time
import cv2
import numpy as np
import random
import utils

from gym_stratego.envs import StrategoEnv

import gym_stratego
print("gym_stratego.__file__: ", gym_stratego.__file__)

env = gym.make("stratego-v0")


stratego_labels = utils.create_stratego_labels()

for episode in range(0, 1):
    #observation = env.large_reset()
    observation = env.reset()

    step = 0
    while True:
        #time.sleep(2.0)

        print("step: ", step)

        key_list = list(observation.keys())
        print("key_list: ", key_list)

        #unit_info = observation['unit_info']
        #print("unit_info: ", unit_info)

        battle_field = observation['battle_field']
        print("battle_field.shape: ", battle_field.shape)

        current_turn = observation["current_turn"]
        print("current_turn: ", current_turn)

        possible_actions = observation['possible_actions']
        print("possible_actions: ", possible_actions)

        action = random.choice(possible_actions)

        observation, reward, done, info = env.step(action)
        print("")
        #observation, reward, done, info = env.step_render()

        if done:
            print("done: ", done)
            break

        step += 1

env.close()