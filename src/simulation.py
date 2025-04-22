# simulation.py
from torch import Tensor
import torch
from agent import Agent
from point import Point
from vehicle import Vehicle
from environment import Environment
import random
import numpy as np


class Simulation:
    def __init__(
        self, vehicle: Vehicle, environment: Environment, agent: Agent, dt: float = 0.1
    ):
        """Represents a simulation of a vehicle in an environment. The simulation is represented by a vehicle object, environment object and agent object.

        Args:
            vehicle (Vehicle): Vehicle object representing the vehicle.
            environment (Environment): Environment object representing the environment.
            agent (Agent): Agent object representing the agent.
            dt (float): Time step for the simulation in seconds.
        """
        self.vehicle = vehicle
        self.environment = environment
        self.agent = agent
        self.dt = dt
        self.reset_sim_status()

    def reset_sim_status(self) -> None:
        """Resets the simulation status."""
        self.total_time_steps = 0
        self.vehicle_in_lane = True
        self.vehicle_in_motion = True

    def sim_reset(
        self, longitude: float, latitude: float, dir_angle_offset: float, speed: float
    ) -> None:
        """Resets the simulation vehicle based on the given longitude, latitude, direction angle offset and speed.
        Also resets the simulation statuses.

        Args:
            longitude (float): Longitude of the vehicle placement on the lane from the start [0] to the end [1].
            latitude (float): Latitude of the vehicle placement in the vehicle from left [0] to right [1].
            dir_angle_offset (float): Direction angle offset of the vehicle in radians from the center of the lane.
            speed (float): Speed of the vehicle in miles per hour.
        """
        self.reset_sim_status()
        center_point, heading = self.environment.position_from_coordinates(
            longitude=longitude,
            latitude=latitude,
            angle_offset=dir_angle_offset,
        )
        self.vehicle.vehicle_setup(center_point, heading, speed)
        self.agent.sensors.update_sensors(
            self.vehicle.center_point, self.vehicle.abs_heading
        )

    def sim_random_reset(self, speed_range: list[float] = [20.0, 75.0]):
        longitude = random.uniform(0, 1)
        latitude = random.uniform(0.25, 0.75)
        dir_angle_offset = random.uniform(-np.pi / 5, np.pi / 5)
        speed = random.uniform(speed_range[0], speed_range[1])
        center_point, heading = self.environment.position_from_coordinates(
            longitude=longitude,
            latitude=latitude,
            angle_offset=dir_angle_offset,
        )
        self.vehicle.vehicle_setup(
            center_point=center_point, abs_heading=heading, speed_mph=speed
        )
        self.agent.sensors.update_sensors(
            self.vehicle.center_point, self.vehicle.abs_heading
        )

    def get_state(self) -> Tensor:
        """
        Returns the current state of the simulation, including the vehicle's heading, speed, and the sensor data.
        speed, heading, *sensor_data
        """
        _, sensor_data = self.agent.sensors.sense(self.environment, self.vehicle)
        return torch.tensor(
            [
                self.vehicle.speed_mph
                / self.vehicle.fps_to_mph(self.vehicle.max_speed_fps),
                self.vehicle.abs_heading / np.pi,
                *(torch.tensor(sensor_data) / 200.0),
            ],
            dtype=torch.float32,
        )

    def sim_step(self):
        """Executes a single step in the simulation.
        1. Gets the current state of the simulation.
        2. Gets the action from the agent based on the current state.
        3. Updates the vehicle's position based on the action.
        4. Updates the sensors based on the vehicle's position and heading.
        5. Updates the simulation status.
        6. Gets reward from agent and returns it.
        """
        self.agent.sensors.update_sensors(
            self.vehicle.center_point, self.vehicle.abs_heading
        )

        state = self.get_state()
        action = self.agent.decide(state)

        steering, acceleration = action[0], action[1]

        self.vehicle.update_position(steering, acceleration, self.dt)

        self.agent.sensors.update_sensors(
            self.vehicle.center_point, self.vehicle.abs_heading
        )

        self.update_sim_status()

        reward = self.agent.compute_reward(
            self.get_state(),
            in_lane=self.vehicle_in_lane,
            in_motion=self.vehicle_in_motion,
        )
        # print(f"reward {reward}")
        return state, action, reward

    def update_sim_status(self) -> None:
        """
        Updates the simulation status based on the vehicle's position and heading.
        """
        self.total_time_steps += 1
        self.vehicle_in_lane = self.environment.point_in_lane(self.vehicle.center_point)
        self.vehicle_in_motion = self.vehicle.speed_mph > 0

    def get_sim_status(self) -> tuple[float, bool, bool]:
        """Returns the current simulation status: the total time steps, vehicle in lane status, vehicle in motion status.

        Returns:
            tuple[float, bool, bool]: Tuple of three values representing: total time steps, vehicle in lane status, vehicle in motion status.
        """
        return self.total_time_steps, self.vehicle_in_lane, self.vehicle_in_motion
