'''
Created on 29 jun. 2012

@author: Jeroen Kools

This basic opponent places its army at random, and also select moves randomly.
'''

import gym_stratego.envs.brains.Brain as Brain
from random import shuffle, choice
import gym_stratego.envs.constants as constants

BOARD_WIDTH = constants.BOARD_WIDTH

class Brain(Brain.Brain):
    def __init__(self, game, army, boardwidth=None):
        self.army = army
        self.game = game

        global BOARD_WIDTH
        if boardwidth: BOARD_WIDTH = boardwidth

    def placeArmy(self, armyHeight):
        positions = []

        if self.army.color == "Blue":
            rows = range(armyHeight)
        else:
            rows = range(BOARD_WIDTH - armyHeight, BOARD_WIDTH)

        for row in rows:
            for column in range(BOARD_WIDTH):
                if self.army.getUnit(column, row) == None:
                    positions += [(column, row)]

        positions.reverse()

        shuffle(list(positions))

        for i, unit in enumerate(self.army.army):
            if unit.isOffBoard():
                unit.tag_number = i
                unit.position = positions.pop()

    def findMove(self):
        move = None
        order = range(len(self.army.army))
        shuffle(list(order))

        for i in order:
            if move: break

            unit = self.army.army[i]
            if not unit.canMove or not unit.alive:
                continue

            (col, row) = unit.getPosition()

            if unit.walkFar:
                dist = range(1, BOARD_WIDTH)
                shuffle(list(dist))
            else:
                dist = [1]

            directions = []
            for d in dist:
                north = (col, row - d)
                south = (col, row + d)
                west = (col - d, row)
                east = (col + d, row)

                nw = (col - d, row - d)
                sw = (col - d, row + d)
                ne = (col + d, row - d)
                se = (col + d, row + d)

                directions += [direction for direction in [north, south, west, east] if
                              direction[0] >= 0 and direction[0] < BOARD_WIDTH and
                              direction[1] >= 0 and direction[1] < BOARD_WIDTH and
                              not self.army.getUnit(direction[0], direction[1]) and
                              self.game.legalMove(unit, direction[0], direction[1])]

                if self.game.diagonal:
                    directions += [direction for direction in [nw, sw, ne, se] if
                              direction[0] >= 0 and direction[0] < BOARD_WIDTH and
                              direction[1] >= 0 and direction[1] < BOARD_WIDTH and
                              not self.army.getUnit(direction[0], direction[1]) and
                              self.game.legalMove(unit, direction[0], direction[1])]

            if len(directions) >= 1:
                move = choice(directions)
                return ((col, row), move)

        return (None, None) # no legal move - lost!

    def observe(self, armies):
        pass