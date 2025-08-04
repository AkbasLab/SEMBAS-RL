# simulation.py
from torch import Tensor
import torch
from agent import Agent
from vehicle import Vehicle
from environment import Environment
import random
import numpy as np


class Simulation:
    WP_MIN_DIST_FACTOR = 1.25

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
        self._wp_index = None
        self._wp_dir = None

        self.reset_sim_status()

    def _get_initial_wp_index(self):
        i = self.environment.lane.nearest_neighbor(self.vehicle.center_point)
        lp = self.environment.lane.control_points[i]
        s = lp - self.vehicle.center_point

        is_ahead = self.vehicle.get_direction_vector().dot(s) >= 0

        num_points = len(self.environment.lane.control_points)

        j = (i + 1) % num_points
        lp_nxt_hyp = self.environment.lane.control_points[j]
        s = lp_nxt_hyp - self.vehicle.center_point
        is_fwd = self.vehicle.get_direction_vector().dot(s) >= 0

        if is_ahead:
            index = i
        else:
            index = (i + (1 if is_fwd else -1)) % num_points

        return index, is_fwd

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
        self._wp_index, is_fwd = self._get_initial_wp_index()
        self._wp_dir = 1 if is_fwd else -1

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
        self._wp_index, is_fwd = self._get_initial_wp_index()
        self._wp_dir = 1 if is_fwd else -1

        self.agent.sensors.update_sensors(
            self.vehicle.center_point, self.vehicle.abs_heading
        )

    def get_next_waypoint(self):
        distance = np.linalg.norm(
            self.environment.lane.control_points[self._wp_index]
            - self.vehicle.center_point
        )
        if distance <= self.environment.lane.lane_width * self.WP_MIN_DIST_FACTOR:
            self._wp_index = (self._wp_index + self._wp_dir) % len(
                self.environment.lane.control_points
            )

        return self.environment.lane.control_points[self._wp_index]

    def calc_wp_heading(self, wp: np.ndarray) -> float:
        sp = wp - self.vehicle.center_point

        return np.atan2(sp[1], sp[0])

    def get_state(self) -> Tensor:
        """
        Returns the current state of the simulation, including the vehicle's heading, speed, and the sensor data.
        speed, heading, waypoint_heading, *sensor_data

        Waypoints are the control points of the lane. The vehicle will be guided to these points to help improve
        training to avoid circling behavior.
        """
        _, sensor_data = self.agent.sensors.sense(self.environment, self.vehicle)
        wp = self.get_next_waypoint()
        wp_heading = self.calc_wp_heading(wp)
        return torch.tensor(
            [
                self.vehicle.speed_mph
                / self.vehicle.fps_to_mph(self.vehicle.max_speed_fps),
                self.vehicle.abs_heading / np.pi,
                wp_heading,  # TODO : try with and without WP heading in state vector
                *(np.array(sensor_data) / 200.0),
            ],
            dtype=torch.float32,
        )

    def sim_step(self, debug=False):
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

        if debug:
            print("Prev steer and acc", state[:2])
            print("Steer and acc", steering, acceleration)
            print("Sensors", state[2:])

        self.vehicle.update_position(steering.item(), acceleration.item(), self.dt)

        self.agent.sensors.update_sensors(
            self.vehicle.center_point, self.vehicle.abs_heading
        )

        self.update_sim_status()

        next_state = self.get_state()
        reward = self.agent.compute_reward(
            next_state,
            in_lane=self.vehicle_in_lane,
            in_motion=self.vehicle_in_motion,
        )
        return state, action, reward, next_state

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
