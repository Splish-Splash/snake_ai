import pygame
from game_objects import Board, Snake
from config import *
import numpy as np
from agent import Agent
import time
import matplotlib.pyplot as plt
import torch

pygame.init()
screen = pygame.display.set_mode((NUMBER_OF_CELLS * CELL_WIDTH, NUMBER_OF_CELLS * CELL_WIDTH))
screen.fill((255, 255, 255))

clock = pygame.time.Clock()

moves = {pygame.K_LEFT: 'left', pygame.K_RIGHT: 'right', pygame.K_DOWN: 'down', pygame.K_UP: 'up'}
scores = []
font = pygame.font.SysFont('comicsansms', 20)
mode = 'ai'
str_to_i = {'empty': 0., 'food': 1., 'body': 2., 'head': 3.}
model = Agent(NUMBER_OF_CELLS * NUMBER_OF_CELLS + 1, 4)


# model.net.load_state_dict(torch.load(f'models\\110000_4.83.txt'))
# model.exploration_rate = 0.05
# model.decay1 = False
# model.decay2 = False


def get_state(board: Board, current_ind=0):
    direction_index = board.snake.direction
    direction = [0] * 4
    direction[direction_index] = 1
    direction.append(current_ind)
    direction.extend(board.food_state()[0])
    direction.extend(board.food_state()[1])
    direction.extend(board.head_state()[0])
    direction.extend(board.head_state()[1])
    state = (np.array(board.get_state(), dtype=np.float32), np.array(direction, dtype=np.float32))
    return state


def game_loop():
    board = Board(screen)
    prev_score = 0
    state = get_state(board)
    i = 0
    history = []
    scores = []
    loss_history = []
    pressed_key = 0
    current_ind = 0
    while True:

        score = board.score
        text = font.render(str(score), True, (0, 0, 200), (127, 127, 127))
        if mode == 'ai':
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    pressed_key = min(event.key - 48, 0)
            action = model.act(state)
            history.append(action)
            board.move(action)
            reward = -board.distance_to_food()
            done = 0
            if board.score > prev_score:
                reward = NUMBER_OF_CELLS * NUMBER_OF_CELLS // 2 * board.score
                current_ind = 0

            if board.snake.check_collision() or current_ind > 40:
                reward = -NUMBER_OF_CELLS * NUMBER_OF_CELLS * 2
                done = 1
                scores.append(board.score)
                print(f'mean: {sum(scores[-100:]) / 100} score: {board.score}')
                board = Board(screen)
                if len(loss_history) < 10_000:
                    board = Board(screen, 5)
                elif len(loss_history) < 25_000:
                    board = Board(screen, 4)
                elif len(loss_history) < 50_000:
                    board = Board(screen, 3)
                elif len(loss_history) < 80_000:
                    board = Board(screen, 2)
                elif len(loss_history) < 100_000:
                    board = Board(screen, 1)

                current_ind = 0
            next_state = get_state(board, current_ind)
            model.cache(state, next_state, action, reward, done)
            q, loss = model.learn()
            loss_history.append(loss)
            state = next_state
            prev_score = board.score
            board.draw()
            screen.blit(text, text.get_rect())
            pygame.display.flip()
            current_ind += 1
            i += 1
            if i and i % 2000 == 0:
                # print(model.exploration_rate)
                plt.plot(loss_history)
                # plt.ylim([0, 20])
                plt.show()
                if i and i % 10000 == 0:
                    torch.save(model.net.state_dict(), f'models\\{i}_{sum(scores[-100:]) / 100}.txt')

            if pressed_key > 0:
                clock.tick(pressed_key)
        else:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        board = Board(screen)
                    else:
                        board.move(moves[event.key])

                if board.game_over:
                    score = board.score
                    scores.append(score)
                    # board = Board(screen)
                board.draw()
                screen.blit(text, text.get_rect())
                pygame.display.flip()


if __name__ == '__main__':
    game_loop()
