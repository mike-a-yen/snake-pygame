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

Point = namedtuple('Point', 'x,y')

FONT = pygame.font.SysFont('arial', 25)