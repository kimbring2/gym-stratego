# Introduction
An OpenAI Gym for Stratego board game to benchmark Reinforcement Learning algorithms

# Install
```
$ git clone https://github.com/kimbring2/gym-stratego
$ cd gym-stratego
$ pip install -e .
```

# Example Code
After installing, run below code.
```
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
    while True:
        print("step: ", step)
        #cv2.imshow('battle_field', battle_field)
        #cv2.waitKey(1)

        print("step_phase: ", step_phase)
        print("step: ", step)

        #env.render()
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
        
        battle_field = observation['battle_field'] / 255.0
        red_offboard = observation['red_offboard']
        blue_offboard = observation['blue_offboard']
        movable_units = observation['movable_units']
        clicked_unit = observation['clicked_unit']
        movable_positions = observation['movable_positions']

        if done:
            break

        time.sleep(1)
        
env.close()
```

