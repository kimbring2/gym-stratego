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


for episode in range(0, 1):
    #observation = env.large_reset()
    observation = env.reset()
    # observation.keys():  dict_keys(['unit_info', 'battle_field', 'red_offboard', 'blue_offboard'])

    step = 0
    while True:
        print("step: ", step)

        key_list = list(observation.keys())
        print("key_list: ", key_list)

        #['battle_field', 'red_offboard', 'blue_offboard', 'movable_units', 'clicked_unit', 'movable_positions', 'red_offboard_rank', 'blue_offboard_rank
        movable_units = observation['movable_units']
        movable_positions = observation['movable_positions']

        print("movable_units: ", movable_units)
        print("movable_positions: ", movable_positions)

        for unit in movable_units:
            select_unit = self.get_unit_from_tag(unit)
            (x, y) = select_unit.position

            observation, reward, done, info = env.step((x, y))
            #self.update_screen()

            movable_positions = observation['movable_positions']

            observation, reward, done, info = env.step((x, y))
            #self.update_screen()

            unit_info[unit] = movable_positions

            time.sleep(0.5)

        print("")

        #unit_info = observation['unit_info']
        #print("unit_info: ", unit_info)

        #unit_list = list(unit_info.keys())
        #print("unit_list: ", unit_list)

        #select_unit = random.choice(unit_list)
        #print("select_unit: ", select_unit)

        #movable_position = unit_info[select_unit]
        #print("movable_position: ", movable_position)

        #select_position = random.choice(movable_position)
        #print("select_position: ", select_position)
        
        #observation, reward, done, info = env.large_step(select_unit, select_position[0], select_position[1])
        observation, reward, done, info = env.step_render()

        if done:
            print("done: ", done)
            break

        step += 1

        #time.sleep(0.5)

env.close()