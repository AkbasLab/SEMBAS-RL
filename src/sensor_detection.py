from point import Point
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
    right = (
        right_edge_intersection
        if right_edge_intersection != -1
        else sensor.sensor_length
    )
    left_edge_point, left_edge_intersection = intersections_on_line_segment(
        lane.left_edge, sensor.origin_point, sensor.end_point
    )
    left = (
        left_edge_intersection if left_edge_intersection != -1 else sensor.sensor_length
    )
    pts = [right_edge_point, left_edge_point]
    intersects = [right, left]
    idx = np.argmin(intersects)

    return pts[idx], intersects[idx]


def intersections_on_line_segment(pt_list: list[Point], pt1: Point, pt2: Point):
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
    for i in range(len(pt_list) - 1):
        intersection = line_segment_intersection(pt1, pt2, pt_list[i], pt_list[i + 1])
        # print(f"closest: {closest_distance}")
        if intersection is not None:

            # print(intersection)
            # distance = (intersection - pt1).norm()
            dist = intersection.distanceTo(pt1)
            # print(dist)
            # distance = np.linalg.norm([dist.x, dist.y])
            # distance = np.linalg.norm([p1.x, p1.y])
            # if dist < closest_distance:
            #     closest_distance = dist
            #     closest_point = intersection
            #     print(f"closest: {closest_distance}")
            distances.append(dist)
            points.append(intersection)

    if len(distances) != 0:
        closest_distance = min(distances)
        closest_point = points[distances.index(closest_distance)]
    else:
        closest_point = None
        closest_distance = -1.0
    return closest_point, closest_distance


# Function to check if two line segments intersect
def line_segment_intersection(p1: Point, p2: Point, q1: Point, q2: Point):
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

    if r_cross_s == 0:
        return None  # Parallel or collinear
    t = cross(q_minus_p, s) / r_cross_s
    u = qmp_cross_r / r_cross_s

    intersection_point = None

    if 0 <= t <= 1 and 0 <= u <= 1:
        intersection_point = p1 + t * r

    return intersection_point


def cross(v1: Point, v2: Point) -> float:
    """Calculates the cross product of two vectors."""

    return v1.x * v2.y - v1.y * v2.x
