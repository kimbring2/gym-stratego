# Introduction
An OpenAI Gym for the Python implementaion of the [Stratego board game](https://github.com/JeroenKools/gpfj) to benchmark Reinforcement Learning algorithms. Thank you for the [JeroenKools](https://github.com/JeroenKools).

<img src="images/stratego_logo.png" width="400">

The DeepMind uses the Stratego game as enviroment at the paper titled [Mastering Stratego, the classic game of imperfect information](https://www.deepmind.com/blog/mastering-stratego-the-classic-game-of-imperfect-information). At this paper, they used the algorithm called DeepNash to find the Nash equilibrium when training the agent using Deep Reinformance Learning. 

# Reference
- Custom OpenAI Gym code: https://github.com/vadim0x60/heartpole
- Python2 implementation of Stratego game: https://github.com/JeroenKools/gpfj

# Install
```
$ git clone https://github.com/kimbring2/gym-stratego
$ cd gym-stratego
$ pip install -e .
```

Additionally, you need to install the ```Tkinter``` for your Python. Below is an example of the Python 3.9 version.
```
$ sudo apt-get install python3.9-tk
```

# OpenAI Gym Sequence
Unlike the other game like a Pong, the Stratego has a multipl unit. Therefore, the step process consist of three phase. At the first phase, you should select one of your unit. At the second phase, the selected unit has the bold boundary line. At the third phase you need to decide the possition where the selectec unit of first phase will move to. 

<img src="images/game_rule_1.png" width="600">

At the start of game, player have no information about rank of opponent unut. However, the rank unit that is engaged in battle and survive is revealed to player.

<img src="images/game_rule_2.png" width="600">

Additionally, you can see the the dead unit of player and opponent at the right side panel. 

# Observation and Action
| State |  Format |
| ------------- | ------------- |
| unit_info | The movable units and their movable position. E.g. ```unit_info:  {1: [(8, 8)], 2: [(7, 7)], 5: [(5, 9)], 7: [(2, 9)], 8: [(0, 8), (1, 9)], 9: [(0, 6), (0, 8), (1, 7)], 10: [(8, 8)], 11: [(7, 7), (8, 8)], 12: [(7, 7), (8, 6), (8, 8)], 13: [(7, 7), (8, 6)], 14: [(5, 9)], 17: [(2, 9)], 18: [(0, 8), (1, 7), (1, 9)], 20: [(8, 4), (9, 3), (9, 5)], 22: [(8, 4), (8, 6), (9, 5)], 23: [(8, 6), (9, 5)], 24: [(5, 9)], 27: [(1, 7)]}```  |
| battle_field | Numpy array that has (10, 10, 3) shape. The value of the array is the int(unit.rank * 10) for the player unit and known opponent unit. The 30.0 for unknown opponent unit.|
| red_offboard | Tag number for the dead unit of the player. E.g. ```[10, 7, 6, 6, 6, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 1]``` |
| blue_offboard | Tag number for the dead unit of opponent. E.g. ```[7, 7, 6, 5, 5, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 11, 11, 1]``` |

# Example Code
```
import gym
import time
import cv2
import numpy as np
import random
from gym_stratego.envs import StrategoEnv

env = gym.make("stratego-v0")

for episode in range(0, 10):
    observation = env.large_reset()
    while True:
        # observation.keys():  dict_keys(['unit_info', 'battle_field', 'red_offboard', 'blue_offboard'])
        unit_info = observation["unit_info"]
        battle_field = observation["battle_field"] / 255.0
        red_offboard = observation["red_offboard"]
        blue_offboard = observation["blue_offboard"]

        cv2.imshow('battle_field', battle_field)
        cv2.waitKey(1)

        unit_list = list(unit_info.keys())
        select_unit = random.choice(unit_list)
        select_unit_position = unit_info[select_unit]
        select_position = random.choice(select_unit_position)

        observation, reward, done, info = env.large_step(select_unit, select_position[0], select_position[1])
        if done:
            break

env.close()
```

# Play as human
It is possible to play the game manually. Please change the ```env.step(action)``` part as of code as ```env.step_render()```.

Below is demo video for that.

[![human play](https://img.youtube.com/vi/QlrTqNp1R3U/sddefault.jpg)](https://youtu.be/QlrTqNp1R3U "Play as human video - Click to Watch!")
<strong>Click to Watch!</strong>
