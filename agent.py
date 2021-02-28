from collections import deque
import logging
import random

import numpy as np
import torch

from game import SnakeGameAI
from game.utils import Direction, BLOCK_SIZE, Point
from model import LinearQNet, QTrainer
from plotter import plot_scores

log = logging.getLogger(__file__)
MAX_MEMORY = 100_000
BATCH_SIZE = 1024
LR = 0.001


class Agent:
    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(11, 256, 4)
        self.trainer = QTrainer(self.model, LR, self.gamma)

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
            mini_batch = random.sample(self.memory, BATCH_SIZE) # list of tuples
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
            log.info('Random move')
            move = random.randint(0, len(action) - 1)
        else:
            log.info('Inferred move')
            prediction = self.model(
                torch.tensor(state, dtype=torch.float)
            )
            move = int(prediction.argmax().item())
        action[move] = 1
        return action


def train():
    single_game_scores = []
    mean_game_scores = []
    smoothing = 15
    assert smoothing > 0
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
                agent.model.save('high_score')

            single_game_scores.append(score)
            mean_game_scores.append(sum(single_game_scores[-smoothing:]) / smoothing)

            log.info(f'[Game {agent.n_games}] Score: {score} ({record})')
            plot_scores(single_game_scores, mean_game_scores)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s]: %(message)s')
    train()
    log.info('Done.')
