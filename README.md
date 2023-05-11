# Introduction
An OpenAI Gym for the Python implementaion of the [Stratego board game](https://github.com/JeroenKools/gpfj) to benchmark Reinforcement Learning algorithms. Thank you for the [JeroenKools](https://github.com/JeroenKools).

# Install
```
$ git clone https://github.com/kimbring2/gym-stratego
$ cd gym-stratego
$ pip install -e .
```

# OpenAI Gym sequence

# Observation and Action
| State |  Format |
| ------------- | ------------- |
| battle_field | Numpy array that has (10, 10, 3) shape. The value of array is the int(unit.rank * 10) for player unit and known opponent unit. The 30.0 for unknown opponent unit.|
| red_offboard | Tag number for the dead unit of player. E.g. [10, 7, 6, 6, 6, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 1] |
| blue_offboard | Tag number for the dead unit of opponent. E.g. [7, 7, 6, 5, 5, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 11, 11, 1] |
| movable_units | Tag number for the unit of player which can control for next turn. E.g. [0, 2, 3, 6, 10, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 26] |
| movable_positions | The x, y coordinate of battle field where the selected unit of player can move for next turn. E.g. [(6, 3), (7, 2), (8, 3)] |

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

# Play as human
It is possible to play the game manually. Please change the ```env.step(action)``` part as below one.
```
observation, reward, done, step_phase = env.step_render()
```

Below is demo video for that.
[![human play](https://img.youtube.com/vi/QlrTqNp1R3U/sddefault.jpg)](https://youtu.be/QlrTqNp1R3U "Play as human video - Click to Watch!")
<strong>Click to Watch!</strong>
