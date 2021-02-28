from collections import namedtuple
from enum import Enum

import pygame

pygame.init()

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class Action(Enum):
    STRAIGHT = [1, 0, 0, 0]
    RIGHT = [0, 1, 0, 0]
    LEFT = [0, 0, 1, 0]
    BACK = [0, 0, 0, 1]


DirectionActionMap = {
    Direction.LEFT: {
        Direction.LEFT: Action.STRAIGHT,
        Direction.RIGHT: Action.BACK,
        Direction.UP: Action.RIGHT,
        Direction.DOWN: Action.LEFT
    },
    Direction.RIGHT: {
        Direction.LEFT: Action.BACK,
        Direction.RIGHT: Action.STRAIGHT,
        Direction.UP: Action.LEFT,
        Direction.DOWN: Action.RIGHT
    },
    Direction.DOWN: {
        Direction.LEFT: Action.RIGHT,
        Direction.RIGHT: Action.LEFT,
        Direction.UP: Action.BACK,
        Direction.DOWN: Action.STRAIGHT
    },
    Direction.UP: {
        Direction.LEFT: Action.LEFT,
        Direction.RIGHT: Action.RIGHT,
        Direction.UP: Action.STRAIGHT,
        Direction.DOWN: Action.BACK
    }
}

Point = namedtuple('Point', 'x,y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 50

FONT = pygame.font.SysFont('arial', 25)