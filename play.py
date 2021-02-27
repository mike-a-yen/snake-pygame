import logging
import pygame

from game.game import SnakeGameHuman


def loop(game) -> None:
    game_over = False
    total_reward = 0
    while not game_over:
        reward, game_over, score = game.play_step()
        total_reward += reward
    return total_reward, score


def main() -> None:
    game = SnakeGameHuman()
    reward, score = loop(game)
    print('Gameover.')
    print(f'Final score: {score}')
    print(f'Reward: {reward}')
    pygame.quit()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s]: %(message)s')
    pygame.init()
    main()
