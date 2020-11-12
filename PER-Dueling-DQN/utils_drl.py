from typing import (
    Optional,
)

import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils_types import (
    TensorStack4,
    TorchDevice,
)

from utils_memory import ReplayMemory
from utils_model import Dueling_DQN


class Agent(object):

    def __init__(
            self,
            action_dim: int,
            device: TorchDevice,
            gamma: float,
            seed: int,

            eps_start: float,
            eps_final: float,
            eps_decay: float,

            restore: Optional[str] = None,
    ) -> None:
        self.__action_dim = action_dim
        self.__device = device
        self.__gamma = gamma

        self.__eps_start = eps_start
        self.__eps_final = eps_final
        self.__eps_decay = eps_decay

        self.__eps = eps_start
        self.__r = random.Random()
        self.__r.seed(seed)

        self.policy = Dueling_DQN(action_dim, device).to(device)
        self.target = Dueling_DQN(action_dim, device).to(device)
        if restore is None:
            self.policy.apply(Dueling_DQN.init_weights)
        else:
            self.policy.load_state_dict(torch.load(restore))
        self.target.load_state_dict(self.policy.state_dict())
        self.__optimizer = optim.Adam(
            self.policy.parameters(),
            lr=0.0000625,
            eps=1.5e-4,
        )
        self.target.eval()

    def run(self, state: TensorStack4, training: bool = False) -> int:
        """run suggests an action for the given state."""
        if training:
            self.__eps -= \
                (self.__eps_start - self.__eps_final) / self.__eps_decay
            self.__eps = max(self.__eps, self.__eps_final)

        if self.__r.random() > self.__eps:
            with torch.no_grad():
                return self.policy(state).max(1).indices.item()
        return self.__r.randint(0, self.__action_dim - 1)

    def learn(self, memory: ReplayMemory, batch_size: int) -> float:
        """learn trains the value network via TD-learning."""
        idxs, (state_batch, next_batch, action_batch, reward_batch, done_batch), is_weights = memory.sample(batch_size)

        y_batch = []
        current_Q_batch = self.policy(next_batch).cpu().data.numpy()
        max_action_next = np.argmax(current_Q_batch, axis=1)
        target_Q_batch = self.target(next_batch)
        
        for i in range(batch_size):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                target_Q_value = target_Q_batch[i, max_action_next[i]]
                y_batch.append(reward_batch[i] + self.__gamma * target_Q_value)

        y_batch = torch.stack(y_batch)
        # print(y_batch.shape)

        values = self.policy(state_batch).gather(1, action_batch)
        # print(values.shape)

        abs_error = torch.abs(y_batch - values)
        memory.batch_update(idxs, abs_error)
        # values_next = self.target(next_batch).max(1).values.detach()
        # expected = (self.__gamma * values_next.unsqueeze(1)) * \
        #     (1. - done_batch) + reward_batch

        # loss = F.smooth_l1_loss(values, expected)
        loss = (torch.FloatTensor(is_weights).to(self.__device) * F.mse_loss(values, y_batch)).mean()

        self.__optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.__optimizer.step()

        return loss.item()

    def sync(self) -> None:
        """sync synchronizes the weights from the policy network to the target
        network."""
        self.target.load_state_dict(self.policy.state_dict())

    def save(self, path: str) -> None:
        """save saves the state dict of the policy network."""
        torch.save(self.policy.state_dict(), path)
