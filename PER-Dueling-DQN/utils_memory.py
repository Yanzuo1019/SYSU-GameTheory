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
    data_pointer = 0

    def __init__(self, channels, capacity, device):
        self.capacity = capacity
        self.device = device
        self.tree = np.zeros(2 * capacity - 1)

        self.__m_states = torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8)
        self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)

    def add(self, p, folded_state, action, reward, done):
        tree_idx = self.data_pointer + self.capacity - 1

        self.__m_states[self.data_pointer] = folded_state
        self.__m_actions[self.data_pointer] = action
        self.__m_rewards[self.data_pointer] = reward
        self.__m_dones[self.data_pointer] = done
        self.update(tree_idx, p)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
    
    def update(self, idx, p):
        change = p - self.tree[idx]
        
        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change
    
    def get_leaf(self, v):
        parent_idx = 0

        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
        
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], \
            self.__m_states[data_idx, :4].to(self.device).float(), \
            self.__m_states[data_idx, 1:].to(self.device).float(), \
            self.__m_actions[data_idx].to(self.device), \
            self.__m_rewards[data_idx].to(self.device).float(), \
            self.__m_dones[data_idx].to(self.device)
    
    @property
    def total_p(self):
        return self.tree[0]

class ReplayMemory(object):
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.0

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

        self.entries = 0
        self.tree = SumTree(channels, capacity, device)
        # self.__m_states = torch.zeros(
        #     (capacity, channels, 84, 84), dtype=torch.uint8)
        # self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        # self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        # self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)

    def push(
            self,
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
        self.entries += 1
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, folded_state, action, reward, done)

    def sample(self, batch_size: int):
        # indices = torch.randint(0, high=self.__size, size=(batch_size,))
        # b_state = self.__m_states[indices, :4].to(self.__device).float()
        # b_next = self.__m_states[indices, 1:].to(self.__device).float()
        # b_action = self.__m_actions[indices].to(self.__device)
        # b_reward = self.__m_rewards[indices].to(self.__device).float()
        # b_done = self.__m_dones[indices].to(self.__device).float()
        # return b_state, b_action, b_reward, b_next, b_done

        idxs = np.empty((batch_size,), dtype=np.int32)
        ISWeights = np.empty((batch_size, 1))

        b_state_list = []
        b_next_list = []
        b_action_list = []
        b_reward_list = []
        b_done_list = []

        pri_seg = self.tree.total_p / batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p
        if min_prob == 0:
            min_prob = 0.00001

        for i in range(batch_size):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            
            idx, p, b_state, b_next, b_action, b_reward, b_done = self.tree.get_leaf(v)
            # print(b_state.shape, b_next.shape, b_action.shape, b_reward.shape, b_done.shape)

            idxs[i] = idx
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)

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

        # print(b_state_list.shape, b_next_list.shape, b_action_list.shape, b_reward_list.shape, b_done_list.shape)
        batch = (b_state_list, b_next_list, b_action_list, b_reward_list, b_done_list)

        return idxs, batch, ISWeights
    
    def batch_update(self, idx, error):
        error += self.epsilon
        clipped_error = np.minimum(error.cpu().data.numpy(), self.abs_err_upper)
        ps = np.power(clipped_error, self.alpha)
        for ti, p in zip(idx, ps):
            self.tree.update(ti, p)

    def __len__(self) -> int:
        return self.entries
