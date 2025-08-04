import numpy as np


class Sensor:
    def __init__(
        self, sensor_length: float, angle_offset: float
    ):  # Tested as of 3/29/2025
        """Represents a sensor in the environment. The sensor is represented by an origin point, end point, and sensor length.

        Args:
            sensor_length (float): Length of the sensor in feet.
            angle_offset (float): Angle offset of the sensor, in radians, from the center of the sensor group.
        """
        self.sensor_length = sensor_length
        self.angle_offset = angle_offset

    def update_sensor(
        self, origin_point: np.ndarray, direction_angle: float
    ):  # Tested as of 3/29/2025
        """Updates the sensor based on the given origin point and direction vector.

        Args:
            origin_point (Point): Point object representing the origin of the sensor.
            direction_angle (float): Point object representing the direction vector of the sensor.
        """
        self.origin_point = origin_point
        self.end_point = self.calculate_end_point(direction_angle)

    def calculate_end_point(
        self, direction_angle: float
    ) -> np.ndarray:  # Tested as of 3/29/2025
        """Calculates the end point of the sensor based on the origin point. Direction vector is the direction of the center of the sensor group.
        The sensors actual direction is calculated by adding the angle offset to the direction vector.
        The end point is calculated by adding the new direction vector to the origin point, scaled by the sensor length.

        Args:
            direction_angle (Point): The direction angle of the center of the group of sensors.

        Returns:
            Point: Point object representing the end point of the sensor.
        """
        # Get the new angle of the sensor
        new_angle = direction_angle + self.angle_offset

        # Compute new heading point using the fixed sensor length
        return (
            self.origin_point
            + np.array([np.cos(new_angle), np.sin(new_angle)]) * self.sensor_length
        )
