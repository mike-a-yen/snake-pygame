import logging
import random
import time
import os

import hydra
import pygame
import wandb

from agent import Agent, init_logging
from game.game import SnakeGameHuman


log = logging.getLogger(__file__)


def loop(agent, game, run) -> None:
    gameover = False
    total_reward = 0
    while not gameover:
        state_old = agent.get_state(game)
        action = game.get_user_action()
        reward, gameover, score = game.play_step(action)
        run.log({'game_id': agent.n_games, 'reward': reward, 'step_score': score})
        state = agent.get_state(game)
        agent.remember(state_old, action.value, reward, state, gameover)
        total_reward += reward
    return total_reward, score


def train(agent: Agent) -> None:
    epochs = 5
    for epoch in range(epochs):
        random.shuffle(agent.memory)
        for i in range(0, len(agent.memory), agent.agent_cfg.batch.size):
            batch_end = min(len(agent.memory), i + agent.agent_cfg.batch.size)
            mini_batch = [agent.memory[j] for j in range(i, batch_end)]
            states, actions, rewards, next_states, gameovers = zip(*mini_batch)
            agent.trainer.train_step(states, actions, rewards, next_states, gameovers)


@hydra.main(config_path='config', config_name='config')
def main(cfg) -> None:
    run = init_logging(cfg, job_type='human-play')
    agent = Agent(cfg.agent)
    if cfg.agent.model.initial_weights is not None:
        artifact = run.use_artifact(cfg.agent.model.initial_weights)
        datadir = artifact.download(root=None)
        state_filename = os.path.join(datadir, 'state.pth')
        agent.model.load(state_filename)
    else:
        artifact = wandb.Artifact(
            'human-supervision',
            type='model',
            metadata={
                'n_games': agent.n_games,
                'n_steps': len(agent.memory),
                'high_score': 0
            }
        )

    game = SnakeGameHuman(cfg.game)
    record = 0
    while True:
        for i in range(3):
            log.info(f'Starting in {3-i}')
            time.sleep(1)
        reward, score = loop(agent, game, run)
        run.log({'game_id': agent.n_games, 'game_score': score, 'game_time': game.frame_iteration})
        log.info('Gameover.')
        log.info(f'Final score: {score} Reward: {reward}')
        log.info(f'Memory size: {len(agent.memory)}')
        stop = input('Quit? [y/N] ').lower() == 'y'
        record = max(score, record)
        if stop:
            break
        game.reset()

    run.summary.update({'high_score': record, 'memory_size': len(agent.memory)})

    run_training = input('Train agent? [y/N] ').lower() == 'y'
    if run_training:
        train(agent)
        model_dir = agent.model.save(run.id)
        for filename in model_dir.iterdir():
            artifact.add_file(str(filename))
        artifact.metadata['n_games'] += agent.n_games
        artifact.metadata['n_steps'] += len(agent.memory)
        artifact.metadata['high_score'] = max(record, artifact.metadata['high_score'])
        run.log_artifact(artifact)
        artifact.save()
        log.info(f'Saved model to {model_dir}')
    pygame.quit()


if __name__ == '__main__':
    pygame.init()
    main()
    log.info('Done.')
