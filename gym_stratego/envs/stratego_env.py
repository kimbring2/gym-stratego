import numpy as np
import gym
import pygame
from PIL import Image, ImageTk
from math import sin, pi
import pkgutil

from gym_stratego.envs.Army import Army, Icons
import gym_stratego.envs.explosion
from gym_stratego.envs.constants import *
from gym_stratego.envs.brains import *
import os

BRAINLIST = [module[1] for module in pkgutil.iter_modules(['brains']) if not module[1] == "Brain"]
dirname = os.path.dirname(__file__)


class StrategoEnv(gym.Env):
    def __init__(self):
        size = "Normal"
        self.boardWidth = SIZE_DICT[size][0]
        self.pools = SIZE_DICT[size][1]
        #self.tilePix = SIZE_DICT[size][2]
        self.tilePix = 120
        self.armyHeight = min(4, (self.boardWidth - 2) / 2)
        self.boardsize = self.boardWidth * self.tilePix
        self.diagonal = False

        self.unitIcons = Icons(self.tilePix)

        grassImage = pygame.image.load("%s/%s/%s" % (dirname, TERRAIN_DIR, LAND_TEXTURE))
        DEFAULT_IMAGE_SIZE = (self.boardWidth * self.tilePix, self.boardWidth * self.tilePix)
        self.grass_image = pygame.transform.scale(grassImage, DEFAULT_IMAGE_SIZE)

        waterImage = pygame.image.load("%s/%s/%s" % (dirname, TERRAIN_DIR, WATER_TEXTURE))
        DEFAULT_IMAGE_SIZE = (self.tilePix, self.tilePix)
        self.water_image = pygame.transform.scale(waterImage, DEFAULT_IMAGE_SIZE)

        self.BLACK = (0, 0, 0)
        self.WHITE = (200, 200, 200)

        pygame.init()

        self.my_font = pygame.font.Font(dirname + '/fonts/FreeSansBold.ttf', 16)
        self.MAIN_SCREEN = pygame.display.set_mode((self.boardsize * 2, self.boardsize * 1))
        self.BATTLE_SCREEN = pygame.Surface((self.boardsize, self.boardsize))
        self.RED_SIDE_SCREEN = pygame.Surface((self.boardsize, int(self.boardsize / 2)))
        self.BLUE_SIDE_SCREEN = pygame.Surface((self.boardsize, int(self.boardsize / 2)))

        CLOCK = pygame.time.Clock()

    def is_movable(self, unit):
        """ Return a list of directly adjacent tile coordinates, considering the edge of the board
        and whether or not diagonal movement is enabled."""
        (x, y) = unit.position

        if unit.rank != 11:
            west = self.legalMove(unit, x - 1, y)
            north = self.legalMove(unit, x, y - 1)
            south = self.legalMove(unit, x, y + 1)
            east = self.legalMove(unit, x + 1, y)
        else:
            west = False
            north = False
            south = False
            east = False
  
        movable_direction = [west, north, south, east]
        movable = west or north or south or east

        return movable

    def get_unit_from_tag(self, tag_number):
        for unit in self.redArmy.army:
            if unit.tag_number == tag_number:
                return unit

    def get_movable_positions(self, tag_number):
        unit = self.get_unit_from_tag(tag_number)

        movable_positions = []
        (x_self, y_self) = unit.position
        for x in range(self.boardWidth):
            for y in range(self.boardWidth):
                if x == x_self and y == y_self:
                    continue

                if self.legalMove(unit, x, y):
                    movable_positions.append((x, y))

        return movable_positions

    def observation(self):
        state = np.ones((self.boardWidth, self.boardWidth, 3)) * 255.0

        movable_units = []
        red_offboard = []
        blue_offboard = []

        for unit in self.redArmy.army:
            #print("unit.tag_number:%d, unit.rank: %d" % (unit.tag_number, unit.rank))
            #movable = self.is_movable(unit)

            if unit.isOffBoard() == False:
                x, y = unit.getPosition()
                state[x, y, 0] = 180.0
                state[x, y, 1] = 180.0
                state[x, y, 2] = int(unit.rank * 10)
            else:
                red_offboard.append(unit.rank)

            if unit.isOffBoard() == False:
                if self.is_movable(unit):
                    movable_units.append(unit.tag_number)

        for unit in self.blueArmy.army:
            if unit.isOffBoard() == False and unit.isKnown == False:
                x, y = unit.getPosition()
                state[x, y, 0] = 30.0
                state[x, y, 1] = 30.0
                state[x, y, 2] = 30.0
            elif unit.isOffBoard() == False and unit.isKnown:
                x, y = unit.getPosition()
                state[x, y, 0] = 30.0
                state[x, y, 1] = 30.0
                state[x, y, 2] = int(unit.rank * 10)
            else:
                blue_offboard.append(unit.rank)

        if self.clicked_unit:
            clicked_unit = (self.clicked_unit).tag_number
        else: 
            clicked_unit = -1

        if self.step_phase == 2:
            movable_positions = self.get_movable_positions((self.clicked_unit).tag_number)
        else:
            movable_positions = -1

        observation = {
            'battle_field': state,
            'red_offboard': red_offboard,
            'blue_offboard': blue_offboard,
            'movable_units' : movable_units,
            'clicked_unit': clicked_unit,
            'movable_positions': movable_positions
        }

        return observation
        
    def reset(self):
        self.newGame()

        return self.observation(), self.reward, self.done, self.step_phase
        
    def move_unit(self, x, y):
        unit = self.getUnit(x, y)

        if self.unit_selected == False and unit:
            if unit.color == "Red":
                unit.selected = True
                self.unit_selected = True
                self.clicked_unit = unit
                self.step_phase = 2

                return self.observation(), self.reward, self.done, self.step_phase
        elif self.unit_selected == True and unit:
            if unit.selected == True and unit.color == "Red":
                unit.selected = False
                self.unit_selected = False
                self.clicked_unit = None
                self.step_phase = 1
            else:
                result = self.moveUnit(x, y)
                unit.selected = False
                self.step_phase = 1
                self.unit_selected = False
                self.clicked_unit = None

                return self.observation(), self.reward, self.done, self.step_phase
        elif self.unit_selected == True and self.clicked_unit:
            result = self.moveUnit(x, y)
            self.clicked_unit.selected = False
            self.unit_selected = False
            self.clicked_unit = None
            self.step_phase = 1

    def step(self, action):
        self.update_screen()
        self.move_unit(action[0], action[1])

        return self.observation(), self.reward, self.done, self.step_phase

    def step_render(self):
        while True:
            self.update_screen()

            if self.step_phase == 3:
                self.step_phase = 1
                break
            elif self.step_phase == 2:
                pass

            event_list = pygame.event.get()
            for event in event_list:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x = int(event.pos[0] / self.tilePix)
                    y = int(event.pos[1] / self.tilePix)

                    print("(%d, %d)" % (x, y))
                    self.move_unit(x, y)
                    return self.observation(), self.reward, self.done, self.step_phase

        return self.observation(), self.reward, self.done, self.step_phase

    def newGame(self, event=None):
        self.blueArmy = Army("classical", "Blue", self.boardWidth * self.armyHeight)
        self.redArmy = Army("classical", "Red", self.boardWidth * self.armyHeight)

        tempBrain = randomBrain.Brain(self, self.redArmy, self.boardWidth)
        tempBrain.placeArmy(self.armyHeight)

        self.redBrain = 0
        self.blueBrain = eval("SmartBrain")

        self.braintypes = {"Blue": self.blueBrain, "Red": self.redBrain}
        self.brains = {"Blue": self.braintypes["Blue"].Brain(self, self.blueArmy, self.boardWidth) if self.braintypes["Blue"] else 0,
                       "Red": self.braintypes["Red"].Brain(self, self.redArmy, self.boardWidth) if self.braintypes["Red"] else 0}

        self.brains["Blue"].placeArmy(self.armyHeight)

        self.unit_selected = False
        self.clicked_unit = None

        self.firstMove = "Red"
        self.turn = self.firstMove
        self.won = False
        self.turnNr = 1
        self.difficulty = "Normal"
        self.started = True

        self.done = False
        self.step_phase = 1
        self.reward = (0, 0)

        #self.MAIN_SCREEN.fill(self.BLACK)
        self.RED_SIDE_SCREEN.fill(self.WHITE)
        self.BLUE_SIDE_SCREEN.fill(self.WHITE)
        pygame.draw.line(self.RED_SIDE_SCREEN, (0, 0, 0), (0, 0), (self.boardsize, 0))

    def isPool(self, x, y):
        """Check whether there is a pool at tile (x,y)."""
        # uneven board size + middle row or even board size + middle 2 rows
        if (self.boardWidth % 2 == 1 and y == self.boardWidth / 2) or \
            ((self.boardWidth % 2 == 0) and (y == self.boardWidth / 2 or y == (self.boardWidth / 2) - 1)):
            return sin(2 * pi * (x + .5) / BOARD_WIDTH * (POOLS + 0.5)) < 0

    def getUnit(self, x, y):
        return self.redArmy.getUnit(x, y) or self.blueArmy.getUnit(x, y)

    def moveUnit(self, x, y):
        """Move a unit according to selected unit and clicked destination"""
        if not self.legalMove(self.clicked_unit, x, y):
            print("You can't move there! If you want, you can right click to deselect the currently selected unit.")
            return False

        if self.clicked_unit.color == "Red":
            thisArmy = self.redArmy
        else:
            thisArmy = self.blueArmy

        # Moving more than one tile will "expose" the unit as a scout
        (i, j) = self.clicked_unit.getPosition()
        if abs(i - x) > 1 or abs(j - y) > 1:
            self.clicked_unit.isKnown = True

        # Do move animation
        stepSize = self.tilePix / MOVE_ANIM_STEPS
        dx = x - i
        dy = y - j

        self.clicked_unit.unit_selected = False
        self.clicked_unit.selected = False
        target = self.getUnit(x, y)
        if target:
            if target.color == self.clicked_unit.color:
                print("You can't move there - tile already occupied!")
                return False
            elif target.color == self.clicked_unit.color and not self.started:  # switch units
                    (xold, yold) = self.clicked_unit.getPosition()
                    target.setPosition(xold, yold)
                    self.clicked_unit.setPosition(x, y)
                    self.clicked_unit = None
            else:
                self.attack(self.clicked_unit, target)
                if self.started:
                    self.endTurn()

            return
        else:
            print("Moved %s to (%s, %s)" % (self.clicked_unit, x, y))
            if (abs(self.clicked_unit.position[0] - x) + abs(self.clicked_unit.position[1] - y)) > 1 and self.clicked_unit.hasMovedFar != True:
                if not self.clicked_unit.hasMoved:
                    thisArmy.nrOfKnownMovable += 1
                elif not self.clicked_unit.isKnown:
                    thisArmy.nrOfUnknownMoved -= 1
                    thisArmy.nrOfKnownMovable += 1

                self.clicked_unit.hasMovedFar = True
                for unit in thisArmy.army:
                    if self.clicked_unit == unit:
                        unit.isKnown = True
                        unit.possibleMovableRanks = ["Scout"]
                        unit.possibleUnmovableRanks = []
                    elif "Scout" in unit.possibleMovableRanks:
                        unit.possibleMovableRanks.remove("Scout")
            elif self.clicked_unit.hasMoved != True:
                thisArmy.nrOfUnknownMoved += 1
                self.clicked_unit.hasMoved = True
                for unit in thisArmy.army:
                    if self.clicked_unit == unit:
                        unit.possibleUnmovableRanks = []

            self.clicked_unit.setPosition(x, y)
            self.clicked_unit.hasMoved = True

        if self.started:
            self.endTurn()

    def victory(self, color, noMoves=False):
        """Show the victory/defeat screen"""
        self.won = True
        if color == "Red":
            if noMoves:
                messageTxt = "The enemy army has been immobilized. Congratulations, you win!"
            else:
                messageTxt = "Congratulations! You've captured the enemy flag!"

            self.reward = (1, 0)
        else:
            if noMoves:
                messageTxt = "There are no valid moves left. You lose."
            else:
                messageTxt = "Unfortunately, the enemy has captured your flag. You lose."

            self.reward = (0, 1)

        casualties = len(self.redArmy.army) - self.redArmy.nrAlive()
        print("%s has won the game in %i turns!" % (color, self.turnNr))

        self.done = True

    def attack(self, attacker, defender):
        """Show the outcome of an attack and remove defeated pieces from the board"""

        ########
        if attacker.color == "Red":
            attackerArmy = self.redArmy
            defenderArmy = self.blueArmy
        else:
            attackerArmy = self.blueArmy
            defenderArmy = self.redArmy

        # Only the first time a piece becomes known, the possible ranks are updated:
        if not attacker.isKnown:
            if attacker.hasMoved:
                attackerArmy.nrOfUnknownMoved -= 1

            attacker.hasMoved = True
            attackerArmy.nrOfKnownMovable += 1
            for unit in attackerArmy.army:
                if unit == attacker:
                    attacker.possibleMovableRanks = [attacker.name]
                    attacker.possibleUnmovableRanks = []
                elif attacker.name in unit.possibleMovableRanks: 
                    unit.possibleMovableRanks.remove(attacker.name)

        if defender.canMove and not defender.isKnown:
            if defender.hasMoved:
                defenderArmy.nrOfUnknownMoved -= 1
            defender.hasMoved = True  # Although it not moved, it is known and attacked, so..
            defenderArmy.nrOfKnownMovable += 1
            for unit in defenderArmy.army:
                if unit == defender:
                    defender.possibleMovableRanks = [defender.name]
                    defender.possibleUnmovableRanks = []
                elif defender.name in unit.possibleMovableRanks: 
                    unit.possibleMovableRanks.remove(defender.name)
        elif not defender.isKnown:
            defenderArmy.nrOfKnownUnmovable += 1
            for unit in defenderArmy.army:
                if unit == defender:
                    defender.possibleUnmovableRanks = [defender.name]
                    defender.possibleMovableRanks = []
                elif defender.name in unit.possibleUnmovableRanks: 
                    unit.possibleUnmovableRanks.remove(defender.name)

        ##########
        text = "A %s %s attacked a %s %s. " % (attacker.color, attacker.name, defender.color, defender.name)
        attacker.isKnown = True
        defender.isKnown = True

        if defender.name == "Flag":
            attacker.position = defender.position
            defender.die()
            self.victory(attacker.color)

        elif attacker.canDefuseBomb and defender.name == "Bomb":
            attacker.position = defender.position
            defender.die()
            defenderArmy.nrOfLiving -= 1
            defenderArmy.nrOfKnownUnmovable -= 1
            attacker.justAttacked = True
            text += "The mine was disabled."
            if (abs(attacker.position[0] - self.blueArmy.army[0].position[0]) + abs(attacker.position[1] - self.blueArmy.army[0].position[1]) == 1):
                self.blueArmy.flagIsBombProtected = False
            
            if (abs(attacker.position[0] - self.redArmy.army[0].position[0]) + abs(attacker.position[1] - self.redArmy.army[0].position[1]) == 1):
                self.redArmy.flagIsBombProtected = False
        elif defender.name == "Bomb":
            x, y = defender.getPosition()
            x = (x + .5) * self.tilePix
            y = (y + .5) * self.tilePix

            attackerTag = "u" + str(id(attacker))
            attacker.die()
            # print 'attacker:', attackerTag, self.map.find_withtag(attackerTag)
            text += "\nThe %s was blown to pieces." % attacker.name

            attackerArmy.nrOfLiving -= 1
            attackerArmy.nrOfKnownMovable -= 1
        elif attacker.canKillMarshal and defender.name == "Marshal":
            attacker.position = defender.position
            defenderArmy.nrOfLiving -= 1
            defenderArmy.nrOfMoved -= 1
            defender.die()
            attacker.justAttacked = True
            text += "The marshal has been assassinated."
        elif attacker.rank > defender.rank:
            attacker.position = defender.position
            defenderArmy.nrOfLiving -= 1
            defenderArmy.nrOfMoved -= 1
            defender.die()
            attacker.justAttacked = True
            text += "The %s was defeated." % defender.name
        elif attacker.rank == defender.rank:
            defenderArmy.nrOfLiving -= 1
            defenderArmy.nrOfMoved -= 1
            attackerArmy.nrOfLiving -= 1
            attackerArmy.nrOfMoved -= 1
            attacker.die()
            defender.die()
            text += "Both units died."
        else:
            attackerArmy.nrOfLiving -= 1
            attackerArmy.nrOfMoved -= 1
            attacker.die()
            text += "The %s was defeated." % attacker.name

        attacker.unit_selected = False
        defender.unit_selected = False

        print("text: ", text)

    def otherPlayer(self, color):
        """Return opposite color"""
        if color == "Red": 
            return "Blue"

        return "Red"

    def otherArmy(self, color):
        """Return opposite army"""
        if color == "Red": 
            return self.blueArmy

        return self.redArmy

    def endTurn(self):
        """Switch turn to other player and check for end of game conditions"""
        self.turn = self.otherPlayer(self.turn)
        self.turnNr += 1

        if self.brains[self.turn] and not self.won:  # computer player?
            (oldlocation, move) = self.brains[self.turn].findMove()

            # check if the opponent can move
            if move == None:
                self.victory(self.otherPlayer(self.turn), True)
                return

            unit = self.getUnit(oldlocation[0], oldlocation[1])
            unit.hasMoved = True

            # Do move animation
            stepSize = self.tilePix / MOVE_ANIM_STEPS
            dx = move[0] - oldlocation[0]
            dy = move[1] - oldlocation[1]

            enemy = self.getUnit(move[0], move[1])
            if enemy:
                self.attack(unit, enemy)
            else:
                unit.setPosition(move[0], move[1])

            # check if player can move
            tempBrain = randomBrain.Brain(self, self.redArmy, self.boardWidth)
            playerMove = tempBrain.findMove()
            if playerMove[0] == None:
                self.victory(self.turn, True)
                return

            if self.difficulty == "Easy":
                for unit in self.redArmy.army:
                    if unit.isKnown and random() <= FORGETCHANCEEASY:
                        unit.isKnown = False

            print("%s moves unit at (%s,%s) to (%s,%s)" % (self.turn, oldlocation[0], oldlocation[1], move[0], move[1]))

        self.turn = self.otherPlayer(self.turn)

    def legalMove(self, unit, x, y):
        if self.isPool(x, y):
            return False

        (ux, uy) = unit.position
        dx = abs(ux - x)
        dy = abs(uy - y)

        if x >= self.boardWidth or y >= self.boardWidth or x < 0 or y < 0:
            return False

        if unit.color == "Red" and self.redArmy.getUnit(x, y):
            return False

        #if not self.started:
        #    if y < (self.boardWidth - 4):
        #        return False
        #    return True

        if unit.walkFar:
            if dx != 0 and dy != 0:
                if self.diagonal:
                    if dx != dy:
                        return False
                else:
                    return False

            if (dx + dy) == 0:
                return False

            if dx > 0 and dy == 0:
                x0 = min(x, ux)
                x1 = max(x, ux)
                for i in range(x0 + 1, x1):
                    if self.isPool(i, y) or self.getUnit(i, y):
                        return False
            elif dy > 0 and dx == 0:
                y0 = min(y, uy)
                y1 = max(y, uy)
                for i in range(y0 + 1, y1):
                    if self.isPool(x, i) or self.getUnit(x, i):
                        return False
            else:
                xdir = dx / (x - ux)
                ydir = dy / (y - uy)
                distance = abs(x - ux)
                for i in range(1, distance):
                    if self.isPool(ux + i * xdir, uy + i * ydir) or self.getUnit(ux + i * xdir, uy + i * ydir):
                        return False
        else:
            s = dx + dy
            if self.diagonal:
                if s == 0 or max(dx, dy) > 1:
                    return False
            elif s != 1:
                return False

        return True

    def drawUnit(self, screen, unit, x, y, color=None):
        """Draw unit tile with correct color and image, 3d border etc"""
        if color == None:
            color = RED_PLAYER_COLOR if unit.color == "Red" else BLUE_PLAYER_COLOR

        hilight = SELECTED_RED_PLAYER_COLOR if unit.color == "Red" else SELECTED_BLUE_PLAYER_COLOR

        #print("unit.name: %s, x: %d, y: %d, unit.color: %s" % (unit.name, x, y, unit.color))

        DEFAULT_IMAGE_POSITION = (x * self.tilePix, y * self.tilePix)

        if unit.alive and unit.color == 'Red' :
            if unit.selected:
                pygame.draw.rect(self.BATTLE_SCREEN, hilight, 
                                 pygame.Rect(int(x * self.tilePix), int(y * self.tilePix), int(self.tilePix), int(self.tilePix)), 5)
            else:
                pygame.draw.rect(self.BATTLE_SCREEN, color, 
                                 pygame.Rect(int(x * self.tilePix), int(y * self.tilePix), int(self.tilePix), int(self.tilePix)), 2)
        elif unit.alive and unit.color == 'Blue':
            if unit.isKnown == False:
                pygame.draw.rect(self.BATTLE_SCREEN, color, 
                                pygame.Rect(int(x * self.tilePix), int(y * self.tilePix), int(self.tilePix), int(self.tilePix)))
            else:
                pygame.draw.rect(self.BATTLE_SCREEN, color, 
                                 pygame.Rect(int(x * self.tilePix), int(y * self.tilePix), int(self.tilePix), int(self.tilePix)), 2)

        if unit.color == 'Red':
            screen.blit(self.unitIcons.getIcon(unit.name), DEFAULT_IMAGE_POSITION)

        if (unit.color == 'Blue' and not unit.alive) or unit.isKnown:
            screen.blit(self.unitIcons.getIcon(unit.name), DEFAULT_IMAGE_POSITION)

        if (unit.name != "Bomb" and unit.name != "Flag" and unit.color == 'Red') or unit.isKnown:
            text_surface = self.my_font.render(str(unit.rank), False, (255, 238, 102))
            screen.blit(text_surface, ((x + .1) * self.tilePix, (y + .1) * self.tilePix))

        if not unit.alive:
            pygame.draw.line(screen, (0, 0, 0), (x * self.tilePix, y * self.tilePix), ((x + 1) * self.tilePix, (y + 1) * self.tilePix))
            pygame.draw.line(screen, (0, 0, 0), (x * self.tilePix, (y + 1) * self.tilePix), ((x + 1) * self.tilePix, y * self.tilePix))

    def offBoard(self, x):
        """Return negative coordinates used to indicate off-board position. Avoid zero."""
        return -x - 1

    def drawSidePanels(self):
        """Draw the unplaced units in the sidebar widget."""
        unplacedRed = 0
        for unit in sorted(self.redArmy.army, key=lambda x: x.sortOrder):
            if unit.isOffBoard():
                x = int(unplacedRed % 10)
                y = int(unplacedRed / 10)

                unit.setPosition(self.offBoard(x), self.offBoard(y))
                self.drawUnit(self.RED_SIDE_SCREEN, unit, x, y)
                unplacedRed += 1

        unplacedBlue = 0
        for unit in sorted(self.blueArmy.army, key=lambda x: x.sortOrder):
            if unit.isOffBoard():
                x = int(unplacedBlue % 10)
                y = int(unplacedBlue / 10)
                unit.setPosition(self.offBoard(x), self.offBoard(y))
                self.drawUnit(self.BLUE_SIDE_SCREEN, unit, x, y)
                unplacedBlue += 1

    def update_screen(self):
        blockSize = self.armyHeight # Set the size of the grid block
        self.BATTLE_SCREEN.blit(self.grass_image, (0, 0))

        for x in range(self.boardWidth):
            for y in range(self.boardWidth):
                if self.isPool(x, y):
                    DEFAULT_IMAGE_POSITION = (x * self.tilePix, y * self.tilePix)
                    self.BATTLE_SCREEN.blit(self.water_image, DEFAULT_IMAGE_POSITION)

        for unit in self.redArmy.army:
            if unit.alive:
                (x, y) = unit.getPosition()
                self.drawUnit(self.BATTLE_SCREEN, unit, x, y)

        for unit in self.blueArmy.army:
            if unit.alive:
                (x, y) = unit.getPosition()
                self.drawUnit(self.BATTLE_SCREEN, unit, x, y)

        for i in range(self.boardWidth - 1):
            x = self.tilePix * (i + 1)
            pygame.draw.line(self.BATTLE_SCREEN, (0, 0, 0), (x, 0), (x, self.boardsize))
            pygame.draw.line(self.BATTLE_SCREEN, (0, 0, 0), (0, x), (self.boardsize, x))

        self.drawSidePanels()

        self.MAIN_SCREEN.blit(self.BATTLE_SCREEN, (0, 0))
        self.MAIN_SCREEN.blit(self.BLUE_SIDE_SCREEN, (self.boardsize, 0))
        self.MAIN_SCREEN.blit(self.RED_SIDE_SCREEN, (self.boardsize, int(self.boardsize / 2)))

        pygame.display.update()

    def render(self, mode=None):
        self.update_screen()