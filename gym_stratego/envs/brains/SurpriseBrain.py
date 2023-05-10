'''
Created on 15 jan. 2013

@author: Junuxx

This opponent randomly chooses from one of the better brains, so that the player doesn't know 
beforehand what they're up against.
'''

import gym_stratego.envs.brains.Brain as Brain
import gym_stratego.envs.brains.CarefulBrain as CarefulBrain
import gym_stratego.envs.brains.SmartBrain as SmartBrain
from random import choice

BRAINS = [CarefulBrain, SmartBrain]

class Brain(Brain.Brain):
    def __init__(self, game, army, boardwidth=None):
        self.actualBrain = choice(BRAINS).Brain(game, army, boardwidth)
        # print self.actualBrain

    def placeArmy(self, armyHeight):
        return self.actualBrain.placeArmy(armyHeight)

    def findMove(self):
        return self.actualBrain.findMove()
