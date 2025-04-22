import torch
from point import Point
import numpy as np
import math


class Vehicle:

    mph_to_fps_conversion = 5280 / 3600
    fps_to_mph_conversion = 3600 / 5280

    mph2_to_fps2_conversion = 5280 / (3600**2)
    fps2_to_mph2_conversion = (3600**2) / 5280

    def __init__(
        self,
        vehicle_length_ft: float = 10,
        vehicle_width_ft: float = 6,
        min_speed_mph: float = 0.0,
        max_speed_mph: float = 150.0,
        sec_0_to_60: float = 8.0,
        max_acceleration=None,
    ):  # Tested as of 3/31/2025
        """Initializes a vehicle object with default parameters.
        min_speed and max_speed are in miles per hour. Represents vehicle speed capabilities.
        The length and width parameters represent the vehicles dimensions in feet.
        The heading_offset is how far, in feet, from the center point the heading point will be located.

        Vehicle motion calculations are done in feet and seconds. All passed in values are converted to feet and seconds before calculations are done.
        When retrieving the properties of speed and acceleration, they are converted from feet/second to miles/hour.

        Vehicle breaking and acceleration stats taken from: https://copradar.com/chapts/references/acceleration.html

        Args:
            vehicle_length_ft (float): Length of the vehicle in feet. Defaults to 12.0.
            vehicle_width_ft (float): Width of the vehicle in feet. Defaults to 6.0.
            min_speed (float): Minimum speed capability of the vehicle in miles per hour. Defaults to 150 mph.
            max_speed (float): Maximum speed capability of the vehicle in miles per hour. Defaults to 0 mph.
            sec_0_to_60 (float): Number of seconds it takes the car to accelerate from 0 to 60 mph. Default is 8.0.
        """
        self.body = VehicleBody(
            vehicle_length=vehicle_length_ft, vehicle_width=vehicle_width_ft
        )
        self.min_speed_fps = self.mph_to_fps(min_speed_mph)
        self.max_speed_fps = self.mph_to_fps(max_speed_mph)
        # Calculating the max acceleration based on the number of seconds it takes the car to go from 0 to 60 miles per hour.
        # Acceleration = (Final speed - initial speed) / time
        self.max_acceleration_fps2 = (
            max_acceleration or self.mph_to_fps(60.0) / sec_0_to_60
        )
        self.max_breaking_fps2 = 15.0

    def vehicle_setup(
        self, center_point: Point, abs_heading: float, speed_mph: float
    ):  # Tested as of 3/29/2025
        """Sets up the vehicle based on the given center point, heading angle and speed.

        Args:
            center_point (Point): Point object representing the center of the vehicle. Grid scale is in feet so X, Y should be in reference to feet.
            heading (float): The heading angle of the vehicle, in radians. Represents the global angle the vehicle is at following unit circle.
            speed_mph (float): Speed of the vehicle in miles per hour.
        """
        self.center_point = center_point
        speed_fps = self.mph_to_fps(speed_mph)
        self.speed_fps = np.clip(speed_fps, self.min_speed_fps, self.max_speed_fps)
        self.abs_heading = abs_heading
        self.distance_travelled_ft = 0
        self.acceleration_fps2 = 0
        self.body.build_body(center_point=center_point, turn_angle=abs_heading)

    def vehicle_capabilities_str(self):  # Tested as of 3/31/2025
        """Returns string with the vehicles speed, acceleration, and breaking capabilities."""
        return (
            f"Vehicle Capabilities: "
            f"Speed: {self.fps_to_mph(self.min_speed_fps)} to {self.fps_to_mph(self.max_speed_fps)} mph -- "
            f"Max Acceleration: {self.max_acceleration_fps2} fps^2 = {self.fps2_to_mph2(self.max_acceleration_fps2)} mph^2 -- "
            f"Max Breaking: {self.max_breaking_fps2} fps^2 = {self.fps2_to_mph2(self.max_breaking_fps2)} mph^2"
        )

    def vehicle_state_str(self):  # Tested as of 3/31/2025
        """Returns string with center, heading, speed mph, acceleration mph^2, and distance travelled feet"""
        return (
            f"Center: ({self.center_point.x}, {self.center_point.y}) -- "
            f"Heading: ({self.abs_heading * 180 / np.pi}) deg -- "
            f"Speed: {self.speed_mph} mph -- "
            f"Acceleration: {self.acceleration_mph2} mph^2 -- "
            f"Distance travelled: {self.distance_travelled_ft} feet"
        )

    @property
    def speed_mph(self):  # Tested as of 3/31/2025
        """Speed in Miles Per Hour"""
        return self.fps_to_mph(self.speed_fps)

    @property
    def acceleration_mph2(self):  # Tested as of 3/31/2025
        """Acceleration in Miles Per Hour Squared"""
        return self.fps2_to_mph2(self.acceleration_fps2)

    @property
    def distance_travelled_miles(self):  # Tested as of 3/31/2025
        """Distance Travelled in Miles"""
        return self.distance_travelled_ft / 5280

    def mph_to_fps(self, value_mph: float):  # Tested as of 3/31/2025
        """Converts given mph value to fps"""
        return value_mph * self.mph_to_fps_conversion

    def fps_to_mph(self, value_fps: float):  # Tested as of 3/31/2025
        """Converts given fps value to mph"""
        return value_fps * self.fps_to_mph_conversion

    def mph2_to_fps2(self, value_mph2: float):  # Tested as of 3/31/2025
        """Converts given mph^2 value to fps^2"""
        return value_mph2 * self.mph2_to_fps2_conversion

    def fps2_to_mph2(self, value_fps2: float):  # Tested as of 3/31/2025
        """Converts given fps^2 value to mph^2"""
        return value_fps2 * self.fps2_to_mph2_conversion

    def get_direction_vector(self, angle: np.float32 = None) -> torch.Tensor:
        """Gets the direction vector x, y for a given angle. If angle is None, uses the vehicles heading angle.

        Args:
            angle (float, optional): Angle of the direction vector. Defaults to None.

        Returns:
            torch.Tensor: Tensor containing x, y values of the direction vector
        """
        angle = angle or self.abs_heading
        return torch.tensor([np.cos(float(angle)), np.sin(float(angle))])

    def get_heading_point(self, angle: float = None) -> Point:
        """Returns the point of the center point of the vehicle updated by the directional vector.

        Args:
            angle (float, optional): Direction angle. Defaults to None.

        Returns:
            Point: Point object with x, y values for the point at the center offset by the direction vector.
        """
        angle = angle or self.abs_heading
        heading_direction = np.array(self.get_direction_vector(angle))
        hx = self.center_point.x + heading_direction[0]
        hy = self.center_point.y + heading_direction[1]
        return Point(hx, hy)

    # def calculate_velocity(self) -> torch.Tensor:  # Tested as of 3/29/2025
    #     """Calculates the velocity of the vehicle based on the center point, heading point, and speed.

    #     Returns:
    #         float: Velocity of the vehicle in miles per hour.
    #     """
    #     direction = self.get_direction()
    #     return direction * self.speed_fps

    def update_position(
        self, steering_rad: float, acceleration_mph2: float, dt_sec: float
    ):  # Tested as of 3/31/2025
        """Updates the position of the vehicle based on the steering, acceleration and time step.

        Uses kinematic equations to update speed, center point, heading angle, and acceleration.

        Args:
            steering (float): Steering angle in radians.
            acceleration (float): Acceleration in miles per hour squared.
            dt_sec (float): Time step in seconds.
        """
        # Updating acceleration by clipping by the vehicle max breaking and acceleration capabilities
        self.acceleration_fps2 = np.clip(
            self.mph2_to_fps2(float(acceleration_mph2)),
            -float(self.max_breaking_fps2),
            float(self.max_acceleration_fps2),
        )

        # v = vo + a t
        new_speed = self.speed_fps + (self.acceleration_fps2 * dt_sec)

        # Update speed based on acceleration and clipping based on the vehicles speed capabilities
        new_speed = np.clip(new_speed, self.min_speed_fps, self.max_speed_fps)

        # Apply steering: rotate the direction vector by the steering input
        turn_angle = steering_rad * dt_sec
        # % (2 * np.pi) # TODO  may add precision error
        new_heading = self.abs_heading + turn_angle
        new_direction = self.get_direction_vector(new_heading)

        # Takes into account acceleration over time
        distance = (self.speed_fps * dt_sec) + (
            0.5 * self.acceleration_fps2 * (dt_sec**2)
        )
        # Update position (move center point based on speed)
        cx = self.center_point.x + new_direction[0] * distance
        cy = self.center_point.y + new_direction[1] * distance
        self.center_point = Point(cx, cy)

        # Update heading
        self.abs_heading = new_heading

        # Update speed
        self.speed_fps = new_speed

        # print(self.abs_heading, self.speed_fps, steering_rad, acceleration_mph2)

        # Adding the distance travelled in feet
        self.distance_travelled_ft += distance

        # Rebuilding the body after updating position
        self.body.build_body(self.center_point, self.abs_heading)


class VehicleBody:
    def __init__(self, vehicle_length: float, vehicle_width: float):
        self.length = vehicle_length
        self.width = vehicle_width
        self.set_base_corners()

    def set_base_corners(self):
        half_length = self.length / 2
        half_width = self.width / 2
        self.base_corners = [
            Point(half_length, -half_width),  # Front Left
            Point(half_length, half_width),  # Front Right
            Point(-half_length, half_width),  # Back Right
            Point(-half_length, -half_width),  # Back Left
        ]

    def build_body(self, center_point: Point, turn_angle: float):
        """Creates the vehicle corners based on the center point and turn angle.

        Corners are in the order of: Front left, front right, back right, back left

        Args:
            center_point (float): Center point of the vehicle.
            turn_angle (float): Heading angle of the vehicle.

        """
        self.corners = []
        for c in self.base_corners:
            c = c + center_point
            c = c.rotate_point_by_radians(center_point, turn_angle)
            self.corners.append(c)
