from sensor import Sensor
from lane import Lane
import numpy as np


def get_lane_detection(sensor: Sensor, lane: Lane) -> tuple[np.ndarray, float]:
    """Returns the closest intersection point and distance from sensor line to lane edges."""
    results = [
        intersections_on_line_segment(
            lane.right_edge, sensor.origin_point, sensor.end_point
        ),
        intersections_on_line_segment(
            lane.left_edge, sensor.origin_point, sensor.end_point
        ),
    ]

    # Filter out non-intersections
    valid = [(pt, dist) for pt, dist in results if dist >= 0]

    if not valid:
        return None, sensor.sensor_length

    # Choose the closest valid one
    closest_point, closest_dist = min(valid, key=lambda x: x[1])
    return closest_point, closest_dist


def intersections_on_line_segment(
    pt_list: np.ndarray, pt1: np.ndarray, pt2: np.ndarray
):
    a = pt_list[:-1]
    b = pt_list[1:]
    r = pt2 - pt1
    s = b - a
    q_minus_p = a - pt1

    def cross2d_batch(u, v):
        return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    r_cross_s = cross2d_batch(np.tile(r, (s.shape[0], 1)), s)
    qmp_cross_r = cross2d_batch(q_minus_p, np.tile(r, (q_minus_p.shape[0], 1)))

    parallel_mask = r_cross_s == 0
    valid_mask = ~parallel_mask

    r_cross_s_safe = np.where(valid_mask, r_cross_s, 1)  # avoid div by 0
    t = cross2d_batch(q_minus_p, s) / r_cross_s_safe
    u = qmp_cross_r / r_cross_s_safe

    within_bounds = (0 <= t) & (t <= 1) & (0 <= u) & (u <= 1) & valid_mask

    if not np.any(within_bounds):
        return None, -1.0

    intersection_points = pt1 + (t[:, None] * r)
    valid_points = intersection_points[within_bounds]
    dists = np.linalg.norm(valid_points - pt1, axis=1)
    min_idx = np.argmin(dists)
    return valid_points[min_idx], dists[min_idx]
