import math

from .utils import Point


def distance(a: Point, b: Point, method: str = 'manhattan'):
    dx = a.x - b.x
    dy = a.y - b.y
    if method == 'manhattan':
        d = abs(dx) + abs(dy)
    elif method == 'euclidean':
        d = math.sqrt(dx**2 + dy**2)
    else:
        raise ValueError(f'arg `method` = {method}, invalid choice')
    return d
