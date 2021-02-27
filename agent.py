from collections import deque
import logging
import random

import numpy as np
import torch

from game import SnakeGameAI
from game.utils import Direction, BLOCK_SIZE, Point

log = logging.getLogger(__file__)
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = None  # TODO
        self.trainer = None  # TODO

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),
            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),
            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, gameover):
        self.memory.append(
            (state, action, reward, next_state, gameover)
        )

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_batch = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_batch = self.memory

        states, actions, rewards, next_states, gameovers = zip(*mini_batch)
        self.trainer.train_step(states, actions, rewards, next_states, gameovers)

    def train_short_memory(self, state, action, reward, next_state, gameover):
        self.trainer.train_step(state, action, reward, next_state, gameover)

    def get_action(self, state):
        # random moves: exploration exploitation tradeoff
        self.epsilon = 80 - self.n_games
        action = [0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 4)
            action[move] = 1
        
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        state_old = agent.get_state(game)
        action = agent.get_action(state_old)
        reward, gameover, score = game.play_step(action)
        state = agent.get_state(game)

        agent.train_short_memory(state_old, action, reward, state, gameover)

        agent.remember(state_old, action, reward, state, gameover)

        if gameover:
            agent.n_games += 1
            game.reset()
            agent.train_long_memory()

            if score > record:
                record = score
                # agent.model.save()

        log.info(f'[Game {agent.n_games}] Score: {score} ({record})')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s]: %(message)s')
    train()
    log.info('Done.')
