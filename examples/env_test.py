import gym
import time
import cv2
import numpy as np
import random
from gym_stratego.envs import StrategoEnv

env = gym.make("stratego-v0")

for episode in range(0, 1):
    observation = env.large_reset()

    #observation, reward, done, info = env.reset()
    #time.sleep(5)

    step = 0
    while True:
        print("step: ", step)
        unit_list = list(observation.keys())
        select_unit = random.choice(unit_list)
        select_unit_position = observation[select_unit]
        select_position = random.choice(select_unit_position)

        #observation, reward, done, info = env.step_render()
        observation, reward, done, info = env.large_step(select_unit, select_position[0], select_position[1])

        if done:
            print("done: ", done)
            break

        #time.sleep(5)
        print("")

        '''
        print("battle_field.shape: ", battle_field.shape)
        print("red_offboard: ", red_offboard)
        print("blue_offboard: ", blue_offboard)
        print("movable_units: ", movable_units)
        print("clicked_unit: ", clicked_unit)
        print("movable_positions: ", movable_positions)
        print("reward: ", reward)
        print("done: ", done)
        '''
        #cv2.imshow('battle_field', battle_field)
        #cv2.waitKey(1)
        #print("step_phase: ", step_phase)

        #env.render()
        #observation, reward, done, step_phase = env.step_render()
        '''
        if step_phase == 1:
            select_unit_tag = random.choice(movable_units)
            select_unit = env.get_unit_from_tag(select_unit_tag)
            #print("select_unit: ", select_unit)

            (x, y) = select_unit.position
            observation, reward, done, info = env.step((x, y))
        elif step_phase == 2:
            select_position = random.choice(movable_positions)
            #print("select_position: ", select_position)

            observation, reward, done, info = env.step(select_position)
        
        battle_field = observation['battle_field'] / 255.0
        red_offboard = observation['red_offboard']
        blue_offboard = observation['blue_offboard']
        movable_units = observation['movable_units']
        clicked_unit = observation['clicked_unit']
        movable_positions = observation['movable_positions']
        step_phase = info['step_phase']
        '''

env.close()