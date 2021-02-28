import logging
import random
from typing import Optional

import numpy as np
import pygame

from .utils import *


log = logging.getLogger(__name__)


class SnakeGameBase:
    def __init__(self, config) -> None:
        self.config = config
        self.rewards = config.game.rewards
        self.block_size = config.game.block_size
        self.play_speed = config.game.speed
        self.w = config.game.width
        self.h = config.game.height
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.score: int
        self.food: Optional[Point]
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

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, self.block_size, self.block_size))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

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

        # 2. move
        move_action = Action(action)
        self._move(move_action) # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = self.rewards.dead
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = self.rewards.eat
            self._place_food()
        else:
            self.snake.pop()
        reward += self.rewards.move
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
