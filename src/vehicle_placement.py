from lane import Lane
import numpy as np


def lateral_adjustment(latitude: float, angle_offset: float) -> float:
    """Given a latitude and an angle_offset, returns an adjusted angle_offset to take into account if the latitude falls on the edge of the lane.

    Args:
        lateral (float): Value 0 to 1 where 0 is the left edge and 1 is the right edge.
        angle_offset (float): The angle, in radians, the heading should be offset from parallel to the center line.

    Returns:
        float: The angle adjusted for being on the edge. If the latitude is 0 (on the left edge) and the angle is between 0 and pi, angle is multiplied by -1.
        If the latitude is 1 (on the right edge) and the angle offset is between 0 and -pi, angle is multiplied by -1.
    """
    # If on the left edge and angle_offset is pointing outside the lane, reverse the angle_offset
    if latitude == 0 and (0 < angle_offset < np.pi):
        angle_offset *= -1
    # Else on the right edge and angle_offset is pointing outside the lane, reverse the angle_offset
    elif latitude == 1 and (0 > angle_offset > -np.pi):
        angle_offset *= -1
    return angle_offset


def open_loop_adjustment(
    longitude: float, latitude: float, angle_offset: float
) -> float:
    """Adjusts the angle_offset to make sure that it wont be pointing outside the lane.

    Args:
        longitude (float): The longitude position down the length of the lane. 0 is the start, 1 is the end.
        latitude (float): The latitude position from left to right in the lane. 0 is the left edge of the lane, 1 is the right.
        angle_offset (float): The angle offset for the heading point in radians.

    Returns:
        float: The angle_offset adjusted for if its at the start or the end of the lane and its not a closed loop lane.
    """
    if (longitude == 0 or longitude == 1) and latitude == 0:
        if angle_offset > 0:
            angle_offset = 0
        elif angle_offset < (-np.pi / 2):
            angle_offset = -np.pi / 2
    elif (longitude == 0 or longitude == 1) and latitude == 1:
        if angle_offset < 0:
            angle_offset = 0
        elif angle_offset > (np.pi / 2):
            angle_offset = np.pi / 2
    elif longitude == 0 or longitude == 1:
        if np.pi / 2 < angle_offset < np.pi:
            angle_offset = np.pi / 2
        if -np.pi / 2 > angle_offset > -np.pi:
            angle_offset = -np.pi / 2
    return angle_offset


def interpolate_points(points: list[np.ndarray], t: float) -> np.ndarray:
    """Given a list of points and a parameter t, returns the interpolated point at that parameter t.
    The parameter t is a value between 0 and 1, where 0 corresponds to the first point in the list and 1 corresponds to the last point.
    The interpolation is done by calculating the distance along the path defined by the points and finding the corresponding point.

    Args:
        points (list[Point]): List of Point objects representing the points to interpolate.
        t (float): Parameter t representing the position along the lane from 0 to 1.

    Returns:
        Point: The interpolated point at the given parameter t.
    """
    if t <= 0:
        return points[0]
    if t >= 1:
        return points[-1]

    total_dist = sum(np.linalg.norm(b - a) for a, b in zip(points[:-1], points[1:]))

    target_dist = t * total_dist
    acc_dist = 0.0

    for a, b in zip(points[:-1], points[1:]):
        s = b - a
        seg_len = np.linalg.norm(s)
        if acc_dist + seg_len >= target_dist:
            local_t = (target_dist - acc_dist) / seg_len

            return a + s * local_t
        acc_dist += seg_len

    return points[-1]


def get_2d_rotation_matrix(angle: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )


def rotate_point_around_pivot(
    pivot: np.ndarray, point: np.ndarray, angle: float
) -> "np.ndarray":
    """Returns the new point that is rotated around a center point by a given angle.

    Args:
        point (Point): Center point of the rotation.
        rotation_angle (float): Angle, in raidans, to rotate from the center point to get the heading point.
        Positive values rotate to the left, negative values rotate to the right. pi = -pi is facing backwards.

    Returns:
        Point: New heading of the vehicle
    """
    # clip angle to [-pi, pi]
    # angle = np.clip(float(angle), -np.pi, np.pi)

    # Compute the direction vector from (x1, y1) to (x2, y2)
    s = point - pivot

    # Rotate the vector
    rot_matrix = get_2d_rotation_matrix(angle)

    # Apply rotation matrix
    new_s = rot_matrix @ s.flatten()

    # Create the offset point
    return pivot + new_s


def get_direction(
    points: list[np.ndarray], t: float, delta: float = 0.01
) -> np.ndarray:
    """Given a list of points and a parameter t, returns the direction vector at that point.
    The direction vector is calculated by taking the difference between the points at t - delta and t + delta.

    Args:
        points (list[Point]): List of Point objects representing the points to interpolate.
        t (float): Parameter t representing the position along the lane from 0 to 1.
        delta (float, optional): A small value to calculate the direction vector. Defaults to 0.01.

    Returns:
        Point: The direction vector at the given parameter t represented by a point.
    """
    t1 = max(0.0, t - delta)
    t2 = min(1.0, t + delta)
    p1 = interpolate_points(points, t1)
    p2 = interpolate_points(points, t2)
    s = p2 - p1
    norm = np.linalg.norm(s)
    return s / norm if norm != 0 else np.ndarray(0.0, 0.0)


def get_center_point(lane: Lane, longitude: float, latitude: float) -> np.ndarray:
    """Given a lane and a longitude and latitude, returns the center point in the lane at the coordinates provided.

    Args:
        lane (Lane): Lane object representing the lane.
        longitude (float): Longitude distance from 0 to 1 where 0 is the start of the lane and 1 is the end of the lane.
        latitude (float): Lateral distance from 0 to 1 where 0 is the left edge of the lane and 1 is the right edge of the lane.

    Returns:
        Point: The center point in the lane at the coordinates provided.
    """
    # 1. Interpolate the corresponding points along each lane edge
    left_pt = interpolate_points(lane.left_edge, longitude)
    right_pt = interpolate_points(lane.right_edge, longitude)

    # 2. Linearly interpolate across the lane from left to right

    return left_pt + (right_pt - left_pt) * latitude
