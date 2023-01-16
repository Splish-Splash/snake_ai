import copy
import math

import pygame
import random
from typing import List, Tuple
from config import *


class BodyPiece:
    """The part of the body that makes up the snake"""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'({self.x}, {self.y})'


class Snake:
    possible_moves = [0, 1, 0, -1, 0]  # list of possible moves, for example down is [0, 1]
    map_moves = {'down': 0, 'right': 1, 'up': 2, 'left': 3}  # map str_move : index_of_move

    def __init__(self, board, num=0):
        # at the start of the game the head of the snake is in the center
        self.head = BodyPiece(NUMBER_OF_CELLS // 2, NUMBER_OF_CELLS // 2)
        self.body = [self.head]  # , BodyPiece(self.head.x - 1, self.head.y), BodyPiece(self.head.x - 2, self.head.y)]
        # body list for all body pieces
        self.direction = random.randint(0, 3)  # snakes starts with random direction
        self.last = self.body[-1].x, self.body[-1].y
        self.board = board
        self.back_move = False
        for i in range(num):
            self.body.append(BodyPiece(self.body[-1].x - 1, self.body[-1].y))
            self.board[self.body[-1].x][self.body[-1].y] = 'body'

    # this function calls when snake eats food, so after that we append new part of the body
    def new_piece(self):
        self.body.append(BodyPiece(self.last[0], self.last[1]))
        self.board[self.last[0]][self.last[1]] = 'body'

    # check if snake is trying to go backward
    def is_opposite(self, move1, move2):
        return move1 != move2 and (move1 + move2) % 2 == 0

    def make_move(self, move):
        move = self.map_moves[move] if type(move) == str else move
        if self.is_opposite(move, self.direction):
            self.back_move = True
            return
        self.direction = move
        self.last = [self.body[-1].x, self.body[-1].y]
        self.board[self.last[0]][self.last[1]] = 'empty'
        self.body[1:] = self.body[:-1]

        self.body[0] = self.head = \
            BodyPiece(self.head.x + self.possible_moves[move], self.head.y + self.possible_moves[move + 1])
        if not self.check_collision():
            self.board[self.head.x][self.head.y] = 'head'
            if len(self.body) > 1:
                self.board[self.body[1].x][self.body[1].y] = 'body'

    def check_collision(self):
        # collision with borders
        if self.head.x not in range(0, NUMBER_OF_CELLS) or self.head.y not in range(0, NUMBER_OF_CELLS):
            return True
        # collision with body parts
        for piece in self.body[1:]:
            if piece.x == self.head.x and piece.y == self.head.y:
                return True
        return self.back_move

    # coordinates of rectangle of mouth
    def get_mouth(self, i, j):
        if self.direction == 1:
            rect = (CELL_WIDTH * (i + 1) - CELL_WIDTH // 5, CELL_WIDTH * j, CELL_WIDTH // 5, CELL_WIDTH)
        elif self.direction == 3:
            rect = (CELL_WIDTH * i, CELL_WIDTH * j, CELL_WIDTH // 5, CELL_WIDTH)
        elif self.direction == 0:
            rect = (CELL_WIDTH * i, CELL_WIDTH * (j + 1) - CELL_WIDTH // 5, CELL_WIDTH, CELL_WIDTH // 5)
        else:
            rect = (CELL_WIDTH * i, CELL_WIDTH * j, CELL_WIDTH, CELL_WIDTH // 5)
        return rect

    def __repr__(self):
        return f'head: {self.head}, body: {self.body}, tail: {self.last}'


class Board:
    # colors of body parts
    colors = {'empty': (127, 127, 127), 'body': (0, 255, 0), 'head': (30, 200, 50), 'food': (255, 0, 0)}

    def __init__(self, screen, num=0):
        self.board = [['empty'] * NUMBER_OF_CELLS for _ in range(NUMBER_OF_CELLS)]
        self.snake = Snake(self.board, num)
        self.board[self.snake.head.x][self.snake.head.y] = 'head'
        self.screen = screen
        self.create_food()
        self.game_over = False
        self.score = 0
        self.draw()

    def draw(self):
        if self.snake.check_collision():
            return

        for i in range(NUMBER_OF_CELLS):
            for j in range(NUMBER_OF_CELLS):
                color = self.colors[self.board[i][j]]
                if self.board[i][j] == 'food':
                    pygame.draw.rect(self.screen, self.colors['empty'],
                                     (CELL_WIDTH * i, CELL_WIDTH * j,
                                      CELL_WIDTH, CELL_WIDTH))
                    pygame.draw.circle(self.screen, color, (i * CELL_WIDTH + CELL_WIDTH // 2,
                                                            j * CELL_WIDTH + CELL_WIDTH // 2),
                                       CELL_WIDTH // 2)
                else:
                    pygame.draw.rect(self.screen, color, (CELL_WIDTH * i, CELL_WIDTH * j,
                                                          CELL_WIDTH, CELL_WIDTH))
                    if self.board[i][j] == 'head':
                        pygame.draw.rect(self.screen, (10, 255, 10), self.snake.get_mouth(i, j))

    def move(self, move):
        if self.game_over:
            return
        self.snake.make_move(move)
        if self.snake.check_collision():
            self.game_over = True
        if (self.snake.head.x, self.snake.head.y) == self.food:
            self.snake.new_piece()
            self.create_food()
            self.score += 1
            self.draw()
        # print(self.snake)

    def create_food(self):
        # create a list of empty cells
        empty = []
        for i in range(NUMBER_OF_CELLS):
            for j in range(NUMBER_OF_CELLS):
                if self.board[i][j] == 'empty':
                    empty.append((i, j))
        # choose random empty cell and create there food
        food_i, food_j = random.choice(empty)
        self.food = (food_i, food_j)
        self.board[food_i][food_j] = 'food'

    def distance_to_food(self):
        food_x, food_y = self.food
        snake_x, snake_y = self.snake.head.x, self.snake.head.y
        # return math.dist((food_x, food_y), (snake_x, snake_y))
        return abs(food_x - snake_x) + abs(food_y - snake_y)

    def direction_to_food(self):
        x, y = self.snake.head.x, self.snake.head.y
        return [x > self.food[0], y > self.food[1], x < self.food[0], y < self.food[1]]

    # get state of the game (2d array)
    def get_state(self):
        empty_board = [[0] * NUMBER_OF_CELLS for _ in range(NUMBER_OF_CELLS)]
        food_state = copy.deepcopy(empty_board)
        food_state[self.food[0]][self.food[1]] = 1
        body_state = copy.deepcopy(empty_board)
        for piece in self.snake.body[1:]:
            body_state[piece.x][piece.y] = 1
        # head_state = copy.deepcopy(empty_board)
        empty_board[self.snake.head.x][self.snake.head.y] = 1

        return [body_state]  # [body_state, empty_board, food_state]

    def food_state(self):
        x = [0] * NUMBER_OF_CELLS
        x[self.food[0]] = 1
        y = [0] * NUMBER_OF_CELLS
        y[self.food[1]] = 1
        return x, y

    def head_state(self):
        x = [0] * NUMBER_OF_CELLS
        y = [0] * NUMBER_OF_CELLS
        x[self.snake.head.x] = 1
        y[self.snake.head.y] = 1
        return x, y