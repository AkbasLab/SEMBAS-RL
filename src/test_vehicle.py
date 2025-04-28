from vehicle import Vehicle
from point import Point
import numpy as np


def test_going_left():
    v = Vehicle()
    center_point = Point(0, 0)
    heading = np.pi / 2
    speed = 0.0
    v.vehicle_setup(center_point=center_point, abs_heading=heading, speed_mph=speed)
    accel = 0.0
    dt = 1.0
    steering = np.pi / 2  # complete left turn 90 degrees
    v.update_position(steering_rad=steering, acceleration_fps2=accel, dt_sec=dt)
    heading_point = v.get_heading_point()
    assert (
        round(heading_point.x, 6) == -1  # * v.heading_offset_ft
    ), f"Vehicle heading x {heading_point.x} should be -1."
    assert (
        round(heading_point.y, 6) == 0
    ), f"Vehicle heading y: {heading_point.y} should be 0 as there is no movement but a 90 degree turn."
    print("Vehicle Test: Turning right with no movement PASSED.")


def test_going_right():
    v = Vehicle()
    center_point = Point(0, 0)
    heading = np.pi / 2
    speed = 0.0
    v.vehicle_setup(center_point=center_point, abs_heading=heading, speed_mph=speed)
    accel = 0.0
    dt = 1.0
    steering = -np.pi / 2  # complete left turn 90 degrees
    v.update_position(steering_rad=steering, acceleration_fps2=accel, dt_sec=dt)
    heading_point = v.get_heading_point()
    assert (
        round(heading_point.x, 6) == 1
    ), f"Vehicle heading x {v.heading_point.x} should be -1."
    assert (
        round(heading_point.y, 6) == 0
    ), f"Vehicle heading y: {v.heading_point.y} should be 0 as there is no movement but a 90 degree turn."
    print("Vehicle Test: Turning left with no movement PASSED.")


def test_going_straight():
    v = Vehicle()
    center_point = Point(0, 0)
    heading = np.pi / 2
    speed = 0.0
    v.vehicle_setup(center_point=center_point, abs_heading=heading, speed_mph=speed)
    # 0 to 60 mph in 8 seconds should be: 27000 mph
    accel = 27000.0
    dt = 8.0
    v.update_position(steering_rad=0.0, acceleration_fps2=accel, dt_sec=dt)
    # print(v.vehicle_state_str())
    assert (
        round(v.center_point.x, 6) == 0
    ), "Vehicle center point x-coordinate should stay the same since vehicle is going straight."
    assert (
        v.center_point.y >= 351.999 and v.center_point.y <= 352.1
    ), f"Vehicle center point y-coordinate {v.center_point.y} should have changed to 352."
    assert v.acceleration_fps2 == 11.0, "27000 mph^2 is 11.0 fps^2"
    assert (
        v.speed_mph >= 59.9 and v.speed_mph <= 60.0
    ), "Speed should be roughly 60.0 mph at 27000.0 mph^2 over 8 seconds"
    print("Vehicle Test: Vehicle going staight with acceleration > 0 PASSED.")


def test_breaking():
    v = Vehicle()
    center_point = Point(0, 0)
    heading = np.pi / 2
    speed = 25.0
    v.vehicle_setup(center_point=center_point, abs_heading=heading, speed_mph=speed)
    accel = -1000
    dt = 5.0
    v.update_position(steering_rad=0.0, acceleration_fps2=accel, dt_sec=dt)
    # print(v.vehicle_state_str())
    assert v.speed_mph < speed, "Vehicle should have slowed down."
    print("Vehicle Test: Vehicle breaking PASSED.")


def test_no_movement():
    v = Vehicle()
    center_point = Point(0, 0)
    heading = np.pi / 2
    speed = 0
    v.vehicle_setup(center_point=center_point, abs_heading=heading, speed_mph=speed)
    acceleration = 0.0
    steering = 0.0
    dt = 1.0
    v.update_position(steering, acceleration, dt)
    assert (
        v.center_point.x == 0
    ), f"Vehicle center point x-coordinate does not match expected value {0}."
    assert (
        v.center_point.y == 0
    ), "Vehicle center point y-coordinate does not match expected value."
    print("Vehicle Test: No movement PASSED.")


def run_tests():
    test_no_movement()
    test_going_straight()
    test_going_left()
    test_going_right()
    test_breaking()
    print("Vehicle Test: All tests PASSED.\n")


if __name__ == "__main__":
    run_tests()
