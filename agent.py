import random
import torch
from network import SnakeNet
from collections import deque
import numpy as np
from config import *


class Agent:  # agent that plays the game
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = SnakeNet(self.state_dim, self.action_dim).float()

        self.exploration_rate = 1.  # start from full random moves
        self.exploration_rate_decay = 0.9999  # 0.99999975
        self.exploration_rate_min = 0.05
        self.decay1 = True
        self.decay2 = True
        self.curr_step = 0

        self.memory = deque(maxlen=100_000)
        self.batch_size = 512
        self.learn_every = 4  # every 4 steps we will update net.target
        self.gamma = 0.9

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def act(self, state):
        """
        :param state of the game
        :return: action index
        """
        # print(self.exploration_rate)
        if random.random() < self.exploration_rate:
            action_idx = random.randint(0, self.action_dim - 1)
            direction = np.argmax(state[1])
            while direction != action_idx and (direction + action_idx) % 2 == 0:
                action_idx = random.randint(0, self.action_dim - 1)
        else:
            state1 = torch.tensor(state[0]).unsqueeze(0)
            state2 = torch.tensor(state[1]).unsqueeze(0)
            state = (state1, state2)
            activation_values = self.net(state, model='online')
            action_idx = torch.argmax(activation_values)

        self.exploration_rate *= self.exploration_rate_decay
        if self.decay1 and 0.1 < self.exploration_rate < 0.2:
            self.decay1 = False
            self.exploration_rate /= 0.999
        if self.decay2 and 0.05 < self.exploration_rate < 0.1:
            self.decay2 = False
            self.exploration_rate /= 0.999
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1

        return action_idx

    def cache(self, state, next_state, action, reward, done):
        # state = torch.tensor(state)
        # next_state = torch.tensor(next_state)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        done = torch.tensor(done)

        self.memory.append((torch.tensor(state[0]), torch.tensor(state[1]),
                            torch.tensor(next_state[0]), torch.tensor(next_state[1]),
                            action, reward, done,))

    def recall(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        batch = random.sample(self.memory, min(batch_size, len(self.memory)))
        state1, state2, next_state1, next_state2, action, reward, done = map(torch.stack, zip(*batch))
        state = [state1, state2]
        next_state = [state1, state2]
        return state, next_state, action, reward, done

    def td_estimate(self, state, action):
        current_Q = self.net(state, model='online')[np.arange(0, min(self.batch_size, action.shape[0])), action]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, dim=1)
        next_Q = self.net(next_state, model='target')[
            np.arange(0, min(self.batch_size, best_action.shape[0])), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target_conv.load_state_dict(self.net.online_conv.state_dict())
        self.net.target_linear.load_state_dict(self.net.online_linear.state_dict())

    def learn(self, num=None):
        if self.curr_step % self.learn_every == 0 or num is None:
            self.sync_Q_target()
        if num is None:
            state, next_state, action, reward, done = self.recall()
        else:
            state, next_state, action, reward, done = self.recall(1)
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)
        return td_est.mean().item(), loss
