import graphics
from point import Point
from sensor_array import SensorArray
from sensor import Sensor
from lane import Lane
from environment import Environment
from vehicle import Vehicle
import math


def test_calc_end_point():
    s = Sensor(sensor_length=50, angle_offset=math.pi / 2)
    s.update_sensor(Point(200, 200), math.pi / 2)
    assert s.end_point.x == 150, "Sensor end point X does not match expected value."
    assert s.end_point.y == 200, "Sensor end point Y does not mathc expected value."
    print("Sensor Test: End point calculation PASSED.")
    # graphics.plot_sensor(s)
    # ax = plt.gca()
    # ax.plot([200, 200], [200, 210], 'g-')
    # graphics.show([100, 300], [100, 300])


def test_angle_offsets():
    sa = SensorArray(num_sensors=5, sensor_length=50, sensor_angle_spread=math.pi)
    assert all(
        sensor.sensor_length == 50 for sensor in sa.sensors
    ), "Sensors length was not set correctly."
    assert (
        sa.sensors[0].angle_offset == math.pi / 2
    ), "Angle offset for first sensor to the left not correct."
    assert (
        sa.sensors[2].angle_offset == 0.0
    ), "Angle offset for the middle sensor is not correct"
    assert (
        sa.sensors[4].angle_offset == -math.pi / 2
    ), "Angle offset for the last sensor to the right not correct."
    print("Sensor Test: Angle offsets PASSED.")


def test_sense():
    sa = SensorArray(num_sensors=5, sensor_length=50, sensor_angle_spread=math.pi)
    lane = Lane([Point(50, 50), Point(150, 50)], lane_width=12, closed_loop=False)
    env = Environment(lane=lane)
    vehicle = Vehicle()
    center_point = Point(50, 50)
    heading = 0.0
    vehicle.vehicle_setup(
        center_point=center_point, abs_heading=heading, speed_mph=25.0
    )
    sa.update_sensors(center_point, vehicle.abs_heading)
    points, distances = sa.sense(env, vehicle=vehicle)
    assert distances[0] == 6.0, "First sensor distance should be 6.0."
    assert distances[4] == 6.0, "Last sensor distance should be 6.0."
    assert distances[2] == -1.0, "Middle sensor distance should be -1.0."
    # print(distances)
    graphics.plot_environment(env)
    graphics.plot_vehicle(vehicle)
    graphics.plot_sensors(sa)
    graphics.plot_sensor_detections(points, distances)
    graphics.show_without_pause("Sensor data", [0, 200], [0, 200])

    print("Sensor Test: Sensing PASSED.")


def run_tests():
    test_calc_end_point()
    test_angle_offsets()
    test_sense()
    print("Sensor Test: All tests PASSED.\n")


if __name__ == "__main__":
    run_tests()
    # test_sense()
