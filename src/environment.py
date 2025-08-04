# environment.py
import torch
from lane import Lane
import numpy as np
import vehicle_placement as VP


class Environment:
    def __init__(self, lane: Lane):
        """Represents the environment in which the agent operates. The environment is represented by a lane object.

        Args:
            lane (Lane): Lane object representing the environment.
        """
        self.lane = lane

    def set_lane(self, lane: Lane):
        """Sets the lane of the environment.

        Args:
            lane (Lane): Lane object representing the environment.
        """
        self.lane = lane

    def point_in_lane(self, point: np.ndarray) -> bool:
        """Checks if a given point is within the lane.

        Args:
            point (Point): Point object representing the point to check.

        Returns:
            bool: True if the point is within the lane, False otherwise.
        """
        center_dist, _, _ = self.point_position_in_lane(point)
        return center_dist >= 0

    def point_position_in_lane(self, point: np.ndarray) -> tuple[float, float, float]:
        """Calculates the position of a given point in the lane.

        Parameters:
            point (Point): The point to check.

        Returns:
            tuple(float, float, float):
            * float: The distance the point is from the center line. Positive if inside the lane boundaries, negative if outside.
            * float: Distance to the left edge of the lane. Positive if the point is to the left of the center line, negative if to the right.
            * float: Distance to the right edge of the lane. Positive if the point is to the right of the center line, negative if to the left.

        Distances to each edge eith the values being [left_edge, right_edge] the following are true...
            * [-, +] indicates the point is right of the center line.
            * [+, -] indicates the point is left of the center line.
            * [+, +] indicates the point is on the center line.
        """

        center_distance, nearest_index = calc_distance(point, self.lane.center_line)
        classification = abs(center_distance)
        if (
            nearest_index == len(self.lane.center_line) - 1
            and not self.lane.closed_loop
        ):
            in_lane = False
        else:
            in_lane = classification <= (self.lane.lane_width / 2)

        left_dist = abs(calc_distance(point, self.lane.left_edge)[0])
        right_dist = abs(calc_distance(point, self.lane.right_edge)[0])

        if right_dist > left_dist:
            right_dist *= -1
        elif left_dist > right_dist:
            left_dist *= -1

        if not in_lane:
            classification *= -1

        return classification, left_dist, right_dist

    def position_from_coordinates(
        self,
        longitude: float,
        latitude: float,
        angle_offset: float,
    ) -> tuple[np.ndarray, float]:
        """Given a longitude and latitude, both from 0 to 1, returns a point inside the lane longitudinal distance along the lane and lateral distance between the lane edges.

        Args:
            longitude (float): Longitude distance from 0 to 1 where 0 is the start of the lane and 1 is the end of the lane.
            latitude (float): Lateral distance from 0 to 1 where 0 is the left edge of the lane and 1 is the right edge of the lane.
            angle_offset (float): Angle offset, in radians, from the center line of the lane to calculate the heading point.
        Returns:
            Point: The center point in the lane at the coordinates provided.
            float: The heading angle from the center point at the angle_offset from parallel to the center line.
        """
        # Ensuring both are from 0 to 1
        longitude = np.clip(longitude, 0.0, 1.0)
        latitude = np.clip(latitude, 0.0, 1.0)

        # center_point, direction_vector = self.get_lane_position_and_heading(longitude=longitude, latitude=latitude, angle_offset=angle_offset)
        center_point = VP.get_center_point(
            lane=self.lane, longitude=longitude, latitude=latitude
        )

        # Adjust the angle offset based on the lane's closed loop status
        angle_offset = VP.lateral_adjustment(latitude, angle_offset)
        if not self.lane.closed_loop:
            angle_offset = VP.open_loop_adjustment(longitude, latitude, angle_offset)

        lane_direction = VP.get_direction(self.lane.center_line, longitude)

        heading = np.atan2(lane_direction[1], lane_direction[0]) + angle_offset

        return center_point, heading


############ Functions to calculate distance between points and lines ################


def determine_distance(
    point: np.ndarray, segment_pt1: np.ndarray, segment_pt2: np.ndarray
):
    """
    Compute the perpendicular distance of point3 from the line segment defined by segment_pt1 and segment_pt2.

    Args:
        point (Point): The point to measure distance from the line (x3, y3).
        segment_pt1 (Point): First point defining the line (x1, y1).
        segment_pt2 (Point): Second point defining the line (x2, y2).


    Returns:
        float: The perpendicular distance from point3 to the line segment.
        Point: The closest point on the line segment to point3.
    """

    # Compute the line direction vector
    sl = segment_pt2 - segment_pt1
    sp = point - segment_pt1

    # Compute the projection of point3 onto the line
    t = (
        np.linalg.norm(sp - sl * np.dot(sp, sl) / np.dot(sl, sl))
        if np.linalg.norm(sl) != 0
        else 0
    )

    # Clamp t to the range [0,1] to ensure the closest point is on the segment
    t = max(0, min(1, t))

    # Compute the closest point on the segment
    closest = segment_pt1 + sl * t

    # Compute the perpendicular distance
    distance = np.linalg.norm(closest - point)

    return distance


def closest_points(point: np.ndarray, curve_pts: list[np.ndarray]):
    """Calculates the distance between the given point and the given list of points.

    Args:
        point (Point): Point to measure distance to
        curve_pts (list[Point]): List of points to determine closest to the given point.

    Returns:
        distance (float): The distance the point is from the closest point in the given list of points.
        nearest_points (list[Point]): List of the two closest points to the given point.
    """
    curve_pts = np.vstack(curve_pts)  # (pt, x|y)

    # Calcualates all distances
    distances = np.linalg.norm(curve_pts - point, axis=1)
    # The index of the nearest point
    nearest_idx = np.argmin(distances)
    # Getting next closest distance
    distances[nearest_idx] = np.inf
    sectond_neared_idx = np.argmin(distances)
    # Getting the two closest points
    pt1 = curve_pts[nearest_idx]
    pt2 = curve_pts[sectond_neared_idx]

    return [pt1, pt2], nearest_idx


def calc_distance(point: np.ndarray, curve_pts: list[np.ndarray]):
    """Calculates the distance between the given point and the given list of points.

    Args:
        point (Point): point to measure distance to
        curve_pts (list[Point]): List of points to determine closest to the given point.

    Returns:
        tuple(float, int):
        * float: The distance the point is from the closest point in the given list of points.
        * int: Index of the closest point in the given list of points.
    """

    closest_pts, nearest_idx = closest_points(point, curve_pts)
    distance = determine_distance(point, closest_pts[0], closest_pts[1])

    return round(distance, 5), nearest_idx
