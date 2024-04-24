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

# Observation and Action
<img src="images/observation_action.png" width="700">

# Observation of off-board unit 
<img src="images/red_blue_offboard.png" width="700">

# Example Code
```
import gym
import time
import cv2
import numpy as np
import random

from gym_stratego.envs import StrategoEnv
import gym_stratego

env = gym.make("stratego-v0")

for episode in range(0, 100):
    observation = env.reset()

    while True:
        battle_field = observation['battle_field']
        current_turn = observation["current_turn"]
        possible_actions = observation['possible_actions']
        red_offboard = observation['red_offboard']
        blue_offboard = observation['blue_offboard']
        action = random.choice(possible_actions)
        observation, reward, done, info = env.step(action)
        if done:
            break

env.close()
```

# Play as human
It is possible to play the game manually. Please change the ```env.step(action)``` part as of code as ```env.step_render()```.

Below is a demo video for that.

[![human play](https://img.youtube.com/vi/QlrTqNp1R3U/sddefault.jpg)](https://youtu.be/QlrTqNp1R3U "Play as human video - Click to Watch!")
<strong>Click to Watch!</strong>

```
import gym
import time
import cv2
import numpy as np
import random

from gym_stratego.envs import StrategoEnv
import gym_stratego

env = gym.make("stratego-v0")

for episode in range(0, 100):
    observation = env.reset()

    while True:
        observation, reward, done, info = env.step_render()

        battle_field = observation['battle_field']
        blue_offboard = observation['blue_offboard']
        movable_units = observation['movable_units']
        clicked_unit = observation['clicked_unit']
        movable_positions = observation['movable_positions']
        red_offboard_rank = observation['red_offboard_rank']
        blue_offboard_rank = observation['blue_offboard_rank']
        if done:
            break

env.close()
```

It is possible to play the game manually. Please change the ```env.step(action)``` part as of code as ```env.step_render()```.

Below is the demo video for that.

[![human play](https://img.youtube.com/vi/QlrTqNp1R3U/sddefault.jpg)](https://youtu.be/QlrTqNp1R3U "Play as human video - Click to Watch!")
<strong>Click to Watch!</strong>
