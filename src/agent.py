# agent.py
from torch import Tensor
from sensor_array import SensorArray
from environment import Environment
from vehicle import Vehicle
import torch
import os
import carlos_logging


class Agent:
    """Parent class for the agent. The agent is responsible for making decisions based on sensor data and the environment.
    The agent is represented by a sensor array and a decision-making function."""

    def __init__(self, sensor_array: SensorArray):
        self.sensors = sensor_array

    def decide(self, state):
        """throws an error if not implemented in child class."""
        raise NotImplementedError("decide() method not implemented in child class.")

    def compute_reward(self, state: Tensor, in_lane: bool, in_motion: bool) -> float:
        """Computes the reward based on the state of the vehicle and the environment."""
        raise NotImplementedError(
            "compute_reward() method not implemented in child class."
        )

    def sense(self, env: Environment, vehicle: Vehicle):
        """Senses the environment using the sensors in the array. The sensors are updated based on the vehicle's position and heading.
        Args:
            env (Environment): Environment object representing the environment.
            vehicle (Vehicle): Vehicle object representing the vehicle.
        """
        sensor_data = self.sensor_array.sense(env, vehicle)
        return sensor_data

    def save(self, dir_path="./checkpoints", tag="latest"):
        os.makedirs(dir_path, exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.model.actor.state_dict(),
                "critic_state_dict": self.model.critic.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
            },
            os.path.join(dir_path, f"agent_{tag}.pt"),
        )
        carlos_logging.log_message(
            f"Saved model checkpoint to {dir_path}/agent_{tag}.pt"
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.model.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
        carlos_logging.log_message(f"Loaded model checkpoint from {path}")


class SimpleAgent(Agent):
    """Simple agent that makes decisions based on the sensor data."""

    def __init__(self, sensor_array: SensorArray):
        super().__init__(sensor_array)

    def decide(self, state):
        """Makes a decision based on the sensor data."""
        # Implement your decision-making logic here
        return 1.0, 1.0

    def compute_reward(self, state, in_lane: bool, in_motion: bool) -> float:
        return 1.0 if in_lane and in_motion else -1.0
