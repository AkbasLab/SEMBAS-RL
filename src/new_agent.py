# summer_agent.py
from agent import Agent
from sensor_array import SensorArray
from environment import Environment
from vehicle import Vehicle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import carlos_logging
import random
from collections import deque


class ActorNetwork(nn.Module):
    """Actor network for the agent. Maps states to actions."""

    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # print(list(x[-1]))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        # Output between -1 and 1 for steering and acceleration actions
        x = torch.tanh(self.fc3(x))
        return x


class CriticNetwork(nn.Module):
    """Critic network for the agent. Maps states and actions to value estimates."""

    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    """Memory buffer for experience replay."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            torch.stack(state),
            (
                torch.stack(action)
                if isinstance(action[0], torch.Tensor)
                else torch.tensor(action)
            ),
            torch.tensor(reward, dtype=torch.float32),
            torch.stack(next_state),
            torch.tensor(done, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class NewAgent(Agent):
    """Reinforcement learning agent using Actor-Critic architecture."""

    WP_MAX_ANGLE = 0.9 * np.pi

    def __init__(
        self,
        sensor_array: SensorArray,
        obs_dim,
        action_dim=2,
        max_sense_dist=200,
        max_speed=75,
        max_accel=5.0,
        max_turn_rate=np.pi * 2,
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.001,
        buffer_size=10000,
        batch_size=256,
        exploration_noise=0.0,
        min_exploration_noise=0.1,
        max_exploration_noise=0.3,
        lr_schedule: list[tuple[float, tuple[float, float]]] = None,
        debug=False,
    ):
        """
        Initialize the Summer Agent.

        Args:
            sensor_array: Array of sensors for environment perception
            obs_dim: Dimension of the observation space
            action_dim: Dimension of the action space
            max_accel: Maximum acceleration
            lr_actor: Learning rate for the actor network
            lr_critic: Learning rate for the critic network
            gamma: Discount factor
            tau: Soft update parameter
            buffer_size: Size of the replay buffer
            batch_size: Batch size for training
            exploration_noise: Standard deviation of exploration noise
            lr_schedule: (% completion, (actor LR, critic LR))
        """
        super().__init__(sensor_array)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_sense_dist = max_sense_dist
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.max_turn_rate = max_turn_rate
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.min_exploration_noise = min_exploration_noise
        self.max_exploration_noise = max_exploration_noise
        self.exploration_noise = exploration_noise
        self.lr_schedule = (
            sorted(lr_schedule, key=lambda x: x[0]) if lr_schedule else None
        )
        self.lr_index = 0

        self.debug = debug

        # Initialize actor and critic networks
        self.actor = ActorNetwork(obs_dim, action_dim)
        self.critic = CriticNetwork(obs_dim, action_dim)

        # Initialize target networks with the same weights
        self.actor_target = ActorNetwork(obs_dim, action_dim)
        self.critic_target = CriticNetwork(obs_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Set up optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Set up replay buffer
        self.memory = ReplayBuffer(buffer_size)

        # For keeping track of the last state
        self.last_state = None
        self.last_action = None

        self.training = True
        carlos_logging.log_message("NewAgent initialized")

    def update_expl_noise(self, episode: int, max_episodes: int):
        self.exploration_noise = max(
            self.min_exploration_noise,
            self.max_exploration_noise * (1 - episode / max_episodes),
        )

    def update_lr(self, episode: int, max_episodes: int):
        if (
            self.lr_schedule
            and self.lr_index < len(self.lr_schedule)
            and episode > self.lr_schedule[self.lr_index][0] * max_episodes
        ):
            print("debug: updating LR")
            lr_actor, lr_critic = self.lr_schedule[self.lr_index][1]
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
            self.lr_index += 1

    def decide(self, state: Tensor):
        """
        Choose an action based on the current state.

        Args:
            state: Current state of the environment

        Returns:
            action: Action to take (steering, acceleration)
        """
        # Set actor to evaluation mode
        self.actor.eval()

        with torch.no_grad():
            # Get action from actor network
            action = self.actor(state)

            # Add exploration noise during training
            if self.training:
                noise = torch.randn_like(action) * self.exploration_noise
                action = torch.clamp(action + noise, -1.0, 1.0)

        # Set actor back to training mode
        self.actor.train()

        # Save state and action for training
        self.last_state = state.clone()
        self.last_action = action.clone()

        action = action * torch.tensor([self.max_turn_rate, self.max_accel])
        return action

    def compute_reward(self, state: Tensor, in_lane: bool, in_motion: bool) -> float:
        """
        Compute the reward based on the current state and environment conditions.

        Args:
            state: Current state of the environment
            in_lane: Whether the vehicle is in the lane
            in_motion: Whether the vehicle is in motion

        Returns:
            reward: Computed reward value
        """
        # Extract important state information

        state_speed, state_heading, wp_heading = state[:3]
        state_sensor = state[3:]  # Sensor readings

        speed_mph = state_speed.item() * self.max_speed
        heading_abs = state_heading.item() * np.pi
        sensor_data = state_sensor * self.max_sense_dist

        # Base reward for staying in lane
        reward = 0.0

        if not in_lane:
            # Heavy penalty for leaving the lane
            reward -= 50.0
            return torch.tensor(reward)

        if not in_motion:
            # Penalty for stopping
            reward -= 50.0
            return torch.tensor(reward)

        # v1: [cos(hd), sin(hd)], v2: [cos(whd), sin(whd)]

        angle_from_wp = np.acos(
            np.array([np.cos(heading_abs), np.sin(heading_abs)]).dot(
                np.array([np.cos(wp_heading), np.sin(wp_heading)])
            )
        )
        if angle_from_wp > 0.9 * np.pi:
            angle_reward = -10
        else:
            angle_reward = 15 * (1 - angle_from_wp / self.WP_MAX_ANGLE)

        reward += angle_reward

        # Reward for speed - encourage moderate speeds
        speed_reward = 30 * state_speed.item() if speed_mph >= 15 else -10
        reward += speed_reward

        # Reward for staying in the center of the lane
        # Use sensor readings to determine distance from center
        # Assuming sensors are arranged symmetrically with center sensor at index len(sensor_data)//2
        center_index = len(sensor_data) // 2
        # Higher reward for staying in the center
        center_reward = 20.0 * state_sensor[center_index].item()

        reward += center_reward

        # Penalty for being close to the edges
        edge_penalty = 0.0
        min_sensor_reading = torch.min(sensor_data).item()
        # print("min sense", min_sensor_reading)
        if min_sensor_reading < 2:  # If any sensor reading is less than 20 units
            edge_penalty = -30.0 * (1.0 - min_sensor_reading / 2.0)
        reward += edge_penalty

        # Small penalty for extreme steering or acceleration to encourage smooth driving
        if hasattr(self, "last_action") and self.last_action is not None:
            action_penalty = -10.0 * torch.sum(torch.abs(self.last_action))
            reward += action_penalty.item()

        if self.debug:
            print("speed, center dist, edge, action, angle")
            print("Rewards:")
            print(
                speed_reward,
                center_reward.item(),
                edge_penalty,
                action_penalty.item(),
                angle_reward,
            )

            print("Final reward", reward)

            
        return torch.tensor([reward])

    def train_step(self, prev_state, action, reward, next_state, done):
        """
        Train the agent using the provided experience.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            done: Whether the episode is done
        """
        # Store experience in replay buffer
        self.memory.push(prev_state, action, reward, next_state, done)

        # Don't train until we have enough samples
        if len(self.memory) < self.batch_size:
            return None, None

        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q_values = self.critic_target(next_states, next_actions).squeeze()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Calculate critic loss
        current_q_values = self.critic(states, actions).squeeze()
        critic_loss = F.mse_loss(current_q_values, target_q_values)

        # Update critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        # Update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return actor_loss.item(), critic_loss.item()

    def _soft_update(self, local_model, target_model):
        """
        Soft update target network parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model: Source network
            target_model: Target network
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, dir_path="./checkpoints", tag="latest"):
        """
        Save the model parameters.

        Args:
            dir_path: Directory to save the parameters
            tag: Tag to identify the save file
        """
        import os

        os.makedirs(dir_path, exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_target_state_dict": self.actor_target.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            },
            os.path.join(dir_path, f"agent_{tag}.pt"),
        )
        carlos_logging.log_message(
            f"Saved model checkpoint to {dir_path}/agent_{tag}.pt"
        )

    def load(self, path):
        """
        Load model parameters.

        Args:
            path: Path to the saved parameters
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        carlos_logging.log_message(f"Loaded model checkpoint from {path}")
