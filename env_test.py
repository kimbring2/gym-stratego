import gym
import time
import cv2
import numpy as np
import random

env = gym.make("gym_stratego.envs:stratego-v0")

for episode in range(0, 10000):
    observation, reward, done, step_phase = env.reset()
    battle_field = observation['battle_field']  / 255.0
    red_offboard = observation['red_offboard']
    blue_offboard = observation['blue_offboard']
    movable_units = observation['movable_units']
    clicked_unit = observation['clicked_unit']
    movable_positions = observation['movable_positions']

    print("battle_field: ", battle_field)
    print("red_offboard: ", red_offboard)
    print("blue_offboard: ", blue_offboard)
    print("movable_units: ", movable_units)
    print("clicked_unit: ", clicked_unit)
    print("movable_positions: ", movable_positions)

    step = 0
    while True:
        print("step: ", step)
        cv2.imshow('battle_field', battle_field)
        cv2.waitKey(1)

        print("step_phase: ", step_phase)

        #env.render()
        observation, reward, done, step_phase = env.step_render()

        '''
        if step_phase == 1:
            select_unit_tag = random.choice(movable_units)
            select_unit = env.get_unit_from_tag(select_unit_tag)
            print("select_unit: ", select_unit)

            (x, y) = select_unit.position
            observation, reward, done, step_phase = env.step((x, y))
        elif step_phase == 2:
            select_position = random.choice(movable_positions)
            print("select_position: ", select_position)
            observation, reward, done, step_phase = env.step(select_position)
        '''
        
        battle_field = observation['battle_field'] / 255.0
        red_offboard = observation['red_offboard']
        blue_offboard = observation['blue_offboard']
        movable_units = observation['movable_units']
        clicked_unit = observation['clicked_unit']
        movable_positions = observation['movable_positions']

        print("battle_field.shape: ", battle_field.shape)
        print("red_offboard: ", red_offboard)
        print("blue_offboard: ", blue_offboard)
        print("movable_units: ", movable_units)
        print("clicked_unit: ", clicked_unit)
        print("movable_positions: ", movable_positions)
        print("reward: ", reward)
        print("done: ", done)

        if done:
            break

        #time.sleep(1)
        print("")

env.close()