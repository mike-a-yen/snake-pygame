from collections import deque
import logging
import random
from typing import List

import hydra
import numpy as np
import torch

from game import SnakeGameAI
from game.utils import Direction, BLOCK_SIZE, Point
from model import LinearQNet, QTrainer
from plotter import plot_scores

log = logging.getLogger(__file__)
MAX_MEMORY = 100_000


class Agent:
    def __init__(self, agent_cfg) -> None:
        self.n_games = 0
        self.agent_cfg = agent_cfg
        self.epsilon = agent_cfg.epsilon  # randomness
        self.random_until = agent_cfg.random_until
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(agent_cfg.model)
        self.trainer = QTrainer(self.model, agent_cfg.lr, agent_cfg.gamma)

    def get_state(self, game):
        head = game.snake[0]
        last_actions = self.get_previous_actions(self.agent_cfg.state.lookback)
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
            game.food.y > game.head.y,  # food down,
            # distance from food
            (game.food.x - game.head.x) / game.w,
            (game.food.y - game.head.y) / game.h
        ]
        state += [direction for action in last_actions for direction in action]
        assert len(state) == self.model.input_size
        return np.array(state, dtype=int)

    def get_previous_actions(self, n: int) -> List[List[int]]:
        """
        Get a list of the previous integer encoded actions
        """
        default_action = [0 for _ in range(self.model.output_size)]
        actions = [default_action for _ in range(n)]
        for i in range(min(n, len(self.memory))):
            step = self.memory[-(i+1)]
            step_action = step[1]
            actions[i] = step_action
        return actions

    def remember(self, state, action, reward, next_state, gameover):
        self.memory.append(
            (state, action, reward, next_state, gameover)
        )

    def train_long_memory(self):
        if len(self.memory) > self.agent_cfg.batch_size:
            mini_batch = random.sample(self.memory, self.agent_cfg.batch_size) # list of tuples
        else:
            mini_batch = self.memory

        states, actions, rewards, next_states, gameovers = zip(*mini_batch)
        self.trainer.train_step(states, actions, rewards, next_states, gameovers)

    def train_short_memory(self, state, action, reward, next_state, gameover):
        self.trainer.train_step(state, action, reward, next_state, gameover)

    def get_action(self, state):
        # random moves: exploration exploitation tradeoff
        # self.epsilon = 80 - self.n_games
        action = [0 for _ in range(self.model.output_size)]
        rand_action_thresh = self.epsilon - (self.n_games * self.epsilon / self.random_until)
        if random.random() < rand_action_thresh:
            move = random.randint(0, len(action) - 1)
        else:
            prediction = self.model(
                torch.tensor(state, dtype=torch.float)
            )
            move = int(prediction.argmax().item())
        action[move] = 1
        return action


@hydra.main(config_path='config', config_name='config')
def train(cfg) -> None:
    single_game_scores = []
    mean_game_scores = []
    smoothing = 15
    assert smoothing > 0
    record = 0
    agent = Agent(cfg.agent)
    game = SnakeGameAI(cfg)

    while True:
        state_old = agent.get_state(game)
        action = agent.get_action(state_old)
        reward, gameover, score = game.play_step(action)
        state = agent.get_state(game)
        agent.remember(state_old, action, reward, state, gameover)

        agent.train_short_memory(state_old, action, reward, state, gameover)

        if gameover:
            single_game_scores.append(score)
            mean_game_scores.append(sum(single_game_scores[-smoothing:]) / smoothing)
            agent.n_games += 1
            game.reset()
            agent.train_long_memory()

            if score > record:
                record = score
                model_path = agent.model.save('high_score')
                fig = plot_scores(single_game_scores, mean_game_scores)
                fig.savefig(str(model_path / 'training_progress.png'))
            log.info(f'[Game {agent.n_games}] Score: {score} ({record})')
            plot_scores(single_game_scores, mean_game_scores)


if __name__ == '__main__':
    train()
    log.info('Done.')
