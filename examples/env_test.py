import gym
import time
import cv2
import numpy as np
import random

from gym_stratego.envs import StrategoEnv

env = gym.make("stratego-v0")

for episode in range(0, 1):
    observation = env.large_reset()
    while True:
        # observation.keys():  dict_keys(['unit_info', 'battle_field', 'red_offboard', 'blue_offboard'])
        unit_info = observation["unit_info"]
        battle_field = observation["battle_field"] / 255.0
        red_offboard = observation["red_offboard"]
        blue_offboard = observation["blue_offboard"]
        print("unit_info: ", unit_info)
        print("battle_field.shape: ", battle_field.shape)
        print("red_offboard: ", red_offboard)
        print("blue_offboard: ", blue_offboard)

        cv2.imshow('battle_field', battle_field)
        cv2.waitKey(1)

        unit_list = list(unit_info.keys())
        select_unit = random.choice(unit_list)
        select_unit_position = unit_info[select_unit]
        select_position = random.choice(select_unit_position)

        observation, reward, done, info = env.large_step(select_unit, select_position[0], select_position[1])
        if done:
            print("done: ", done)
            break

env.close()