from sensor import Sensor
from lane import Lane
import numpy as np


def get_lane_detection(sensor: Sensor, lane: Lane) -> float:
    """Takes in the edges of the lane and the sensor.
    Returns the closest point and closest distance of the intersection between the lane edges and the sensor line.

    Args:
        sensor (Sensor): Sensor object representing the sensor.
        lane (Lane): Lane object representing the lane.

    Returns:
        tuple(Point, float): Point of the closest intersection and distance to intersection. If no intersection, returns None and -1.0.
    """
    right_edge_point, right_edge_intersection = intersections_on_line_segment(
        lane.right_edge, sensor.origin_point, sensor.end_point
    )
    left_edge_point, left_edge_intersection = intersections_on_line_segment(
        lane.left_edge, sensor.origin_point, sensor.end_point
    )

    closest_point = None
    intersection = None
    if right_edge_intersection == -1.0:
        intersection = left_edge_intersection
        closest_point = left_edge_point if intersection != -1.0 else None
    elif left_edge_intersection == -1.0:
        intersection = right_edge_intersection
        closest_point = right_edge_point
    elif right_edge_intersection <= left_edge_intersection:
        intersection = right_edge_intersection
        closest_point = right_edge_point
    else:
        intersection = left_edge_intersection
        closest_point = left_edge_point
    # if right_edge_intersection == -1.0 and left_edge_intersection == -1.0:
    #     intersection = -1.0
    #     closest_point = None
    # elif right_edge_intersection == -1.0 and left_edge_intersection != -1.0:
    #     intersection = left_edge_intersection
    #     closest_point = left_edge_point
    # elif right_edge_intersection != -1.0 and left_edge_intersection == -1.0:
    #     intersection = right_edge_intersection
    #     closest_point = right_edge_point
    # elif right_edge_intersection <= left_edge_intersection:
    #     intersection = right_edge_intersection
    #     closest_point = right_edge_point
    # else:
    #     intersection = left_edge_intersection
    #     closest_point = left_edge_point

    if intersection == -1:
        intersection = sensor.sensor_length

    return closest_point, intersection


def intersections_on_line_segment(
    pt_list: np.ndarray, pt1: np.ndarray, pt2: np.ndarray
):
    """Takes in a list of points and two points. Returns the intersection points and distances between the two points.

    Args:
        pt_list (list[Point]): List of points representing the line segment.
        pt1 (Point): Start point of the line segment.
        pt2 (Point): End point of the line segment.

    Returns:
        tuple(Point, float): Point and distance to intersection. If no intersection, returns None and -1.0.
    """
    distances = []
    points = []
    # for i in range(len(pt_list) - 1):
    for a, b in zip(pt_list[:-1], pt_list[1:]):
        intersection = line_segment_intersection(pt1, pt2, a, b)
        if intersection is not None:
            dist = np.linalg.norm(pt1 - intersection)
            distances.append(dist)
            points.append(intersection)

    if len(distances) != 0:
        closest_distance = min(distances)
        closest_point = points[distances.index(closest_distance)]
    else:
        closest_point = None
        closest_distance = -1.0

    return closest_point, closest_distance


def cross(a: np.ndarray, b: np.ndarray):
    return a[0] * b[1] - a[1] * b[0]


# Function to check if two line segments intersect
def line_segment_intersection(
    p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray
):
    """Takes in start point and end point of two lines (p1, p2) and (q1, q2).

    Args:
        p1 (Point): Start point of the first line segment.
        p2 (Point): End point of the first line segment.
        q1 (Point): Start point of the second line segment.
        q2 (Point): End point of the second line segment.

    Returns:
        Point: Intersection point if exists, otherwise None.

    """
    r = p2 - p1
    s = q2 - q1
    q_minus_p = q1 - p1
    r_cross_s = cross(r, s)
    qmp_cross_r = cross(q_minus_p, r)
    # qmp_cross_r = cross(q_minus_p, r)

    if r_cross_s == 0:
        return None  # Parallel or collinear
    t = cross(q_minus_p, s) / r_cross_s
    u = qmp_cross_r / r_cross_s

    intersection_point = None

    if 0 <= t <= 1 and 0 <= u <= 1:
        intersection_point = p1 + t * r

    return intersection_point
