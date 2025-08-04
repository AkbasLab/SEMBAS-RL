from environment import Environment
import numpy as np
import sensor_detection as SD
from sensor import Sensor
from vehicle import Vehicle


class SensorArray:
    def __init__(
        self, num_sensors: int, sensor_length: float, sensor_angle_spread: float
    ):  # Tested as of 3/29/2025
        """Represents an array of sensors in the environment. The sensor array is represented by a list of sensor objects.

        Args:
            num_sensors (int): Number of sensors in the array.
            sensor_length (float): Length of each sensor in feet.
            sensor_angle_spread (float): Angle spread of the sensors in radians.
        """
        self.num_sensors = num_sensors
        self.sensor_angle_spread = sensor_angle_spread
        self.setup_sensors(sensor_length)

    def setup_sensors(self, sensor_length: float):  # Tested as of 3/29/2025
        """Sets up the sensors in the array based on the number of sensors, sensor length and angle spread.
        Args:
            sensor_length (float): Length of each sensor in feet.
        """
        angle_max = self.sensor_angle_spread / 2
        angles = np.linspace(angle_max, -angle_max, self.num_sensors)
        self.sensors = [
            Sensor(sensor_length=sensor_length, angle_offset=angle) for angle in angles
        ]

    def update_sensors(
        self, origin_point: np.ndarray, direction_angle: float
    ):  # Tested as of 3/29/2025
        """Updates the sensors in the array based on the given origin point and direction vector.

        Args:
            origin_point (Point): Point object representing the origin of the sensor array.
            direction_angle (float): Angle of the direction of the center of the sensors.
        """

        for sensor in self.sensors:
            sensor.update_sensor(origin_point, direction_angle)

    def sense(self, env: Environment, vehicle: Vehicle):
        """Senses the environment using the sensors in the array.

        Args:
            env (Environment): Environment object representing the environment.
            vehicle (Vehicle): The vehicle the sensors are "on"

        Returns:
            list[float]: List of distances to the nearest obstacle in the direction of each sensor.
        """
        # Update the sensors based on the vehicle's position and heading
        self.update_sensors(
            origin_point=vehicle.center_point,
            direction_angle=vehicle.abs_heading,
        )
        detection_points = []
        detection_distances = []
        for sensor in self.sensors:
            # Calculate the distance to the nearest obstacle in the direction of the sensor
            point, distance = SD.get_lane_detection(sensor=sensor, lane=env.lane)
            detection_points.append(point)
            detection_distances.append(distance)
        return detection_points, detection_distances
