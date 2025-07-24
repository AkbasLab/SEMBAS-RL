from agent import Agent
from sensor_array import SensorArray
import torch


class PresentationAgent(Agent):
    """Simple agent that makes decisions based on the sensor data."""

    def __init__(self, sensor_array: SensorArray):
        super().__init__(sensor_array)

    def decide(self, state):
        """Makes a decision based on the sensor data."""
        # print(state)
        # Implement your decision-making logic here
        # sensor detections, vehicle speed, vehicle heading
        accel = 0.5
        detections = state[2:]
        index = torch.argmax(detections).item()
        # steering = self.sensors.sensors[index].end_point
        steering = self.sensors.sensors[index].angle_offset
        # print(steering)

        return steering, accel

    def compute_reward(self, state, in_lane: bool, in_motion: bool) -> float:
        return 1.0 if in_lane and in_motion else -1.0
