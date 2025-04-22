import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from agent import Agent
from sensor_array import SensorArray
import os
import carlos_logging


class Actor(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Learnable log std

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))

        mean = self.mean(x2)
        std = torch.exp(self.log_std)
        return mean, std


class Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.value(x)


class SummerAgent(Agent):
    def __init__(
        self,
        sensor_array: SensorArray,
        obs_dim: int,
        action_dim: int = 2,
        max_accel=5.0,
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
    ):
        """Agent that uses an actor-critic architecture for decision making.

        Args:
            sensor_array (SensorArray): Sensor array for the agent.
            obs_dim (_type_): Dimension of the observation space.
            action_dim (_type_): Dimension of the action space. Defaults to 2.
            max_accel (float, optional): Maximum acceleration that can be generated. Defaults to 5.0.
            lr_actor (_type_, optional): Learning rate for the actor. Defaults to 1e-4.
            lr_critic (_type_, optional): Learning rate for the critic. Defaults to 1e-3.
            gamma (float, optional): Discount rate for training. Defaults to 0.99.
        """
        super().__init__(sensor_array)
        self.obs_dim = obs_dim
        self.out_dim = action_dim

        self.actor = Actor(obs_dim, action_dim)
        self.critic = Critic(obs_dim)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.max_accel = max_accel
        self.output_scaling = torch.tensor([np.pi, self.max_accel])

    # def decide(self, state: torch.Tensor):
    #     # state = state.reshape(-1, self.obs_dim)
    #     mean, std = self.actor(state)
    #     dist = torch.distributions.Normal(mean, std)
    #     raw_action = dist.rsample()
    #     self.last_log_prob = dist.log_prob(raw_action).sum()
    #     self.last_state = state.clone()
    #     self.last_action = raw_action.clone().detach()

    #     # steering = torch.tanh(raw_action[0]) * np.pi
    #     # accel = torch.sigmoid(raw_action[1]) * self.max_accel
    #     return raw_action * self.output_scaling

    def decide(self, state: torch.Tensor):
        mean, std = self.actor(state)
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()  # This keeps gradients
        self.last_log_prob = dist.log_prob(raw_action).sum()  # Save for training
        self.last_state = state  # No need to clone
        self.last_action = raw_action  # No need to detach

        return raw_action * self.output_scaling

    def compute_reward(self, state, in_lane: bool, in_motion: bool) -> float:
        heading, speed, *sensor_data = state
        reward = 0.0

        # 1. Stay in lane
        reward += 1.0 if in_lane else -2.0

        # 2. Encourage motion
        reward += 0.5 if in_motion else -0.5

        # 3. Encourage higher speed (normalized)
        reward += 0.1 * (speed / 75.0)

        # 4. Penalize proximity to obstacles
        min_sensor = min(sensor_data)
        if min_sensor < 1.0:
            reward -= 1.0 - min_sensor

        # 6. Small time penalty
        reward -= 0.01

        return reward

    def train_step(self, next_state: torch.Tensor, reward, done):
        # next_state = next_state.reshape(-1, self.obs_dim)
        reward_tensor = torch.tensor([reward], dtype=torch.float32)
        done_tensor = torch.tensor([done], dtype=torch.float32)

        value = self.critic(self.last_state)
        with torch.no_grad():
            next_value = self.critic(next_state)
            target = reward_tensor + self.gamma * next_value * (1 - done_tensor)
        advantage = target.detach() - value

        # Update critic
        critic_loss = F.mse_loss(value, target.detach())
        self.critic_optim.zero_grad()
        critic_loss.backward()

        critic_grad_norm = sum(
            p.grad.norm().item() for p in self.critic.parameters() if p.grad is not None
        )
        print(f"Critic grad norm: {critic_grad_norm}")

        self.critic_optim.step()

        # Update actor
        actor_loss = -self.last_log_prob * advantage.detach()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        actor_grad_norm = sum(
            p.grad.norm().item() for p in self.actor.parameters() if p.grad is not None
        )
        print(f"Actor grad norm: {actor_grad_norm}")

    def save(self, dir_path="./checkpoints", tag="latest"):
        os.makedirs(dir_path, exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optim.state_dict(),
                "critic_optimizer_state_dict": self.critic_optim.state_dict(),
            },
            os.path.join(dir_path, f"agent_{tag}.pt"),
        )
        carlos_logging.log_message(
            f"Saved model checkpoint to {dir_path}/agent_{tag}.pt"
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optim.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optim.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        carlos_logging.log_message(f"Loaded model checkpoint from {path}")
