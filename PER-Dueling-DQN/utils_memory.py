from typing import (
    Tuple,
)

import random
import numpy as np
import torch

from utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    TensorStack5,
    TorchDevice,
)

class SumTree:
    write = 0

    def __init__(self, channels, capacity, device):
        self.device = device
        self.capacity = capacity
        self.n_entries = 0

        self.tree = np.zeros(2 * capacity - 1)
        self.__m_states = torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8)
        self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
    
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        return self.tree[0]

    def add(self, p, folded_state, action, reward, done):
        idx = self.write + self.capacity - 1

        self.__m_states[self.write] = folded_state
        self.__m_actions[self.write] = action
        self.__m_rewards[self.write] = reward
        self.__m_dones[self.write] = done
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx, p):
        change = p - self.tree[idx]
        
        self.tree[idx] = p
        self._propagate(idx, change)
    
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (
            idx, 
            self.tree[idx], 
            self.__m_states[dataIdx, :4].to(self.device).float(), 
            self.__m_states[dataIdx, 1:].to(self.device).float(),
            self.__m_actions[dataIdx].to(self.device),
            self.__m_rewards[dataIdx].to(self.device).float(),
            self.__m_dones[dataIdx].to(self.device).float()
        )

class ReplayMemory(object):
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice,
    ) -> None:
        # self.__device = device
        # self.__capacity = capacity
        # self.__size = 0
        # self.__pos = 0

        self.tree = SumTree(channels, capacity, device)
        # self.__m_states = torch.zeros(
        #     (capacity, channels, 84, 84), dtype=torch.uint8)
        # self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        # self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        # self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)

    def __get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def push(
            self,
            error,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
    ) -> None:
        # self.__m_states[self.__pos] = folded_state
        # self.__m_actions[self.__pos, 0] = action
        # self.__m_rewards[self.__pos, 0] = reward
        # self.__m_dones[self.__pos, 0] = done

        # self.__pos = (self.__pos + 1) % self.__capacity
        # self.__size = max(self.__size, self.__pos)
        p = self.__get_priority(error)
        self.tree.add(p, folded_state, action, reward, done)

    def sample(self, batch_size: int):
        # indices = torch.randint(0, high=self.__size, size=(batch_size,))
        # b_state = self.__m_states[indices, :4].to(self.__device).float()
        # b_next = self.__m_states[indices, 1:].to(self.__device).float()
        # b_action = self.__m_actions[indices].to(self.__device)
        # b_reward = self.__m_rewards[indices].to(self.__device).float()
        # b_done = self.__m_dones[indices].to(self.__device).float()
        # return b_state, b_action, b_reward, b_next, b_done
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        b_state_list = []
        b_next_list = []
        b_action_list = []
        b_reward_list = []
        b_done_list = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, b_state, b_next, b_action, b_reward, b_done) = self.tree.get(s)
            idxs.append(idx)
            priorities.append(p)

            b_state_list.append(b_state)
            b_next_list.append(b_next)
            b_action_list.append(b_action)
            b_reward_list.append(b_reward)
            b_done_list.append(b_done)
        
        b_state_list = torch.stack(b_state_list)
        b_next_list = torch.stack(b_next_list)
        b_action_list = torch.stack(b_action_list)
        b_reward_list = torch.stack(b_reward_list)
        b_done_list = torch.stack(b_done_list)

        batch = (b_state_list, b_next_list, b_action_list, b_reward_list, b_done_list)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight
    
    def update(self, idx, error):
        p = self.__get_priority(error)
        self.tree.update(idx, p)

    def __len__(self) -> int:
        return self.tree.n_entries
