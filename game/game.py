import logging
import random
from typing import Optional

import numpy as np
import pygame

from .calculators import distance
from .utils import Action, Direction, FONT, Point
from .colors import *

log = logging.getLogger(__name__)


class SnakeGameBase:
    def __init__(self, game_config) -> None:
        self.config = game_config
        self.rewards = self.config.rewards
        self.block_size = self.config.block_size
        self.play_speed = self.config.speed
        self.w = self.config.width
        self.h = self.config.height
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.score: int
        self.food: Optional[Point]
        self.food_last_eaten: int
        self.frame_iteration: int
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - self.block_size, self.head.y),
                      Point(self.head.x - (2 * self.block_size), self.head.y)]

        self.score = 0
        self.food = None
        self.food_last_eaten = 0
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - self.block_size ) // self.block_size ) * self.block_size
        y = random.randint(0, (self.h - self.block_size ) // self.block_size ) * self.block_size
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self):
        raise NotImplementedError('Must overload in subclass.')

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - self.block_size or pt.x < 0 or pt.y > self.h - self.block_size or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def is_timeout(self):
        return self.frame_iteration - self.food_last_eaten >= self.config.timeout

    def compute_reward(self, delta_distance_to_food) -> float:
        if self.is_collision() or self.is_timeout():
            return self.rewards.dead
        reward = (
            self.rewards.move * delta_distance_to_food / (self.w + self.h)
            + self.rewards.hunger * (self.frame_iteration - self.food_last_eaten)
        )
        if self.head == self.food:
            reward += self.rewards.eat
        return reward

    def _update_ui(self):
        self.display.fill(BLACK)

        snake_colors = SNAKE_COLORS[self.config.snake.color]
        for pt in self.snake:
            pygame.draw.rect(self.display, snake_colors[0], pygame.Rect(pt.x, pt.y, self.block_size, self.block_size))
            pygame.draw.rect(self.display, snake_colors[1], pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size))

        text = FONT.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        text = FONT.render(f'Frame: {self.frame_iteration}', True, WHITE)
        self.display.blit(text, [self.w - 128, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if action == Action.STRAIGHT:#np.array_equal(action, Action.STRAIGHT):
            next_idx = idx
        elif action == Action.RIGHT: # np.array_equal(action, Action.RIGHT):
            next_idx = (idx + 1) % 4
        elif action == Action.LEFT: #np.array_equal(action, Action.LEFT): # [0, 0, 1, 0]
            next_idx = (idx - 1) % 4
        else:
            # go straight rather than backwards, avoid auto lose
            next_idx = idx
            # next_idx = (idx + 2) % 4
        log.debug(f'Move: {self.direction} -> {clock_wise[next_idx]}')
        self.direction = clock_wise[next_idx]

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.UP:
            y -= self.block_size

        self.head = Point(x, y)


class SnakeGameAI(SnakeGameBase):
    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        old_distance_to_food = distance(self.head, self.food, 'manhattan')
        # 2. move
        move_action = Action(action)
        self._move(move_action) # update the head
        self.snake.insert(0, self.head)
        distance_to_food = distance(self.head, self.food, 'manhattan')

        # 3. check if game over
        delta_distance = distance_to_food - old_distance_to_food
        reward = self.compute_reward(delta_distance)
        game_over = False
        if self.is_collision() or self.is_timeout():
            game_over = True
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            self.food_last_eaten = self.frame_iteration
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(self.play_speed)
        # 6. return game over and score
        return reward, game_over, self.score


class SnakeGameHuman(SnakeGameAI):
    def get_user_action(self):
        """
        Get the keyboard input and map it to a model action.
        """
        last_direction = self.direction
        direction = last_direction
        action = Action.STRAIGHT
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    direction = Direction.DOWN
            action = DirectionActionMap[last_direction][direction]
            log.info(f'Action: {action} ({last_direction} -> {direction})')
        return action


    def play_step(self):
        return super().play_step(self.get_user_action())
