import numpy as np
import torch


class Point:
    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def __str__(self):
        return "Point(" + str(self.x) + ", " + str(self.y) + ")"

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def norm(self, p: int = 2) -> float:
        return (self.x**p + self.y**p) ** (1.0 / p)

    def dot(self, other: "Point") -> float:
        return self.x * other.x + self.y * other.y

    def __mul__(self, other: float) -> "Point":
        return Point(other * self.x, other * self.y)

    def __rmul__(self, other: float) -> "Point":
        return self.__mul__(other)

    def __truediv__(self, other: float) -> "Point":
        return self.__mul__(1.0 / other)

    def values(self):
        return [self.x, self.y]

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(self.values())

    def distanceTo(self, other: "Point") -> float:
        return (self - other).norm(p=2)

    def rotate_point_by_radians(self, point: "Point", rotation_angle: float) -> "Point":
        """Returns the new point that is rotated around a center point by a given angle.

        Args:
            point (Point): Center point of the rotation.
            rotation_angle (float): Angle, in raidans, to rotate from the center point to get the heading point.
            Positive values rotate to the left, negative values rotate to the right. pi = -pi is facing backwards.

        Returns:
            Point: New heading of the vehicle
        """
        # clip angle to [-pi, pi]
        rotation_angle = np.clip(float(rotation_angle), -np.pi, np.pi)

        # Compute the direction vector from (x1, y1) to (x2, y2)
        x1, y1 = point.x, point.y
        x2, y2 = self.x, self.y
        dx = x2 - x1
        dy = y2 - y1

        # Rotate the vector
        cos_angle = np.cos(rotation_angle)
        sin_angle = np.sin(rotation_angle)

        # Apply rotation matrix
        new_dx = cos_angle * dx - sin_angle * dy
        new_dy = sin_angle * dx + cos_angle * dy

        # Create the offset point
        offset_x = x1 + new_dx
        offset_y = y1 + new_dy

        return Point(offset_x, offset_y)
