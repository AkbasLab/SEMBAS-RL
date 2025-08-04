from lane import Lane
from environment import Environment
from vehicle import Vehicle
from sensor_array import SensorArray

# import carlos_app as app
import graphics
import math
from sensor import Sensor
import test_vehicle_placement
import test_simulation
import test_environment
import test_vehicle
import test_sensors
import layout_utils as LO


def test_init_lane():
    ctrl_pts, width, closed = LO.load_lane_from_file(
        "./layouts/train_straight_layout_0.txt"
    )
    lane = Lane(control_points=ctrl_pts, lane_width=width, closed_loop=closed)
    assert isinstance(lane, Lane), "Lane initialization failed."
    assert len(lane.control_points) > 0, "Lane control points are empty."
    assert lane.lane_width > 0, "Lane width is not positive."
    assert isinstance(
        lane.closed_loop, bool
    ), "Lane closed loop parameter is not boolean."
    assert (
        lane.control_points[0].x == 2.1387573383383085
    ), "Lane control points do not match expected values."
    assert (
        lane.control_points[0].y == 397.41216641207694
    ), "Lane control points do not match expected values."
    assert (
        lane.closed_loop == False
    ), "Lane closed loop parameter does not match expected value."
    assert lane.lane_width == 12.0, "Lane width does not match expected value."
    print("App Test: Lane initialization test PASSED.")
    return lane  # Return the lane object for further testing


def test_init_environment():
    ctrl_pts, width, closed = LO.load_lane_from_file("./layouts/train_layout_4.txt")
    lane = Lane(control_points=ctrl_pts, lane_width=width, closed_loop=closed)
    assert len(lane.center_line) == len(
        lane.left_edge
    ), f"Center line {len(lane.center_line)} and edges {len(lane.left_edge)} should have the same number of points."
    env = Environment(lane=lane)
    assert env is not None, "Environment initialization failed."
    assert isinstance(env.lane, Lane), "Environment lane is not a Lane object."
    print("App Test: Environment initialization test PASSED.")
    return env  # Return the environment object for further testing


def test_init_vehicle():
    center_point = np.ndarray(10, 10)
    heading = math.pi / 2
    speed = 10.0
    vehicle = Vehicle()
    vehicle.vehicle_setup(
        center_point=center_point, abs_heading=heading, speed_mph=speed
    )
    assert vehicle is not None, "Vehicle initialization failed."
    assert hasattr(vehicle, "center_point"), "Vehicle center point is not set."
    assert hasattr(vehicle, "heading"), "Vehicle heading is not set."
    assert hasattr(vehicle, "speed_fps"), "Vehicle speed is not set."
    assert (
        vehicle.center_point.x == center_point.x
    ), "Vehicle center point x-coordinate does not match."
    assert (
        vehicle.center_point.y == center_point.y
    ), "Vehicle center point y-coordinate does not match."

    assert round(vehicle.speed_mph, 3) == speed, "Vehicle speed does not match."
    assert vehicle.body is not None, "Vehicle body is not built."
    print("App Test: Vehicle initialization test PASSED.")
    return vehicle  # Return the vehicle object for further testing


def test_init_sensor_array():
    """Test function to initialize the sensor array."""
    sensor_array = SensorArray(
        num_sensors=5, sensor_length=50, sensor_angle_spread=math.pi / 2
    )
    assert sensor_array is not None, "Sensor array initialization failed."
    assert (
        sensor_array.num_sensors == 5
    ), "Sensor array number of sensors does not match."
    assert (
        sensor_array.sensor_angle_spread == math.pi / 2
    ), "Sensor array angle spread does not match."
    assert (
        len(sensor_array.sensors) == 5
    ), "Sensor array sensors list length does not match."
    assert all(
        isinstance(sensor, Sensor) for sensor in sensor_array.sensors
    ), "Sensor array sensors are not Sensor objects."
    assert all(
        sensor.sensor_length == 50 for sensor in sensor_array.sensors
    ), "Sensor array sensors length does not match."
    assert (
        sensor_array.sensors[0].angle_offset == math.pi / 4
    ), "Sensor array first sensor angle does not match."
    assert (
        sensor_array.sensors[4].angle_offset == -math.pi / 4
    ), "Sensor array last sensor angle does not match."
    assert (
        sensor_array.sensors[2].angle_offset == 0
    ), "Sensor array middle sensor angle does not match."
    print("App Test: Sensor array initialization test PASSED.")
    return sensor_array  # Return the sensor array object for further testing


def test_sensor():
    s = Sensor(sensor_length=50, angle_offset=-math.pi / 2)
    origin = np.ndarray(0, 0)
    direction = np.ndarray(0, 1)
    s.update_sensor(origin, direction)
    assert (
        s.origin_point == origin
    ), "Sensor origin point does not match expected value."
    assert (
        s.angle_offset == -math.pi / 2
    ), "Sensor angle offset does not match expected value."
    assert (
        s.end_point.x == 50
    ), f"Sensor end point x-coordinate {s.end_point.x} does not match expected value."
    assert (
        s.end_point.y == 0
    ), "Sensor end point y-coordinate does not match expected value."
    print("App Test: Single sensor test PASSED.")


def sensor_tests():
    sensor_array = test_init_sensor_array()
    test_sensor()
    center_point = np.ndarray(0, 0)
    sensor_array.update_sensors(
        center_point, np.ndarray(0, 1)
    )  # Update sensor positions based on vehicle
    assert (
        sensor_array.sensors[0].origin_point == center_point
    ), "Sensor origin point does not match vehicle center point."
    assert (
        sensor_array.sensors[2].end_point.x == 0
    ), "Sensor end point x-coordinate does not match expected value."
    print("App Test: Sensor tests PASSED.")


def test_graphics():
    env = test_init_environment()
    vehicle = test_init_vehicle()
    sensor_array = test_init_sensor_array()
    graphics.plot_environment(env)  # Plot the environment
    graphics.plot_vehicle(vehicle)  # Plot the vehicle
    sensor_array.update_sensors(
        vehicle.center_point, vehicle.abs_heading
    )  # Update sensor positions based on vehicle
    # print(f"center: {vehicle.center_point.values()}, heading: {vehicle.heading_point.values()}")
    # print([s.angle_offset for s in sensor_array.sensors])
    graphics.plot_sensors(sensor_array)
    graphics.show_without_pause(title="Graphics Test")  # Show the plot
    print("App Test: Graphics test NEEDS VISUAL CONFIRMATION.")


#### Main Test Function ####
def initialization_tests():
    test_init_lane()
    test_init_environment()
    test_init_vehicle()
    print("App Test: Initialization tests PASSED.\n")


def run_all_tests():
    test_vehicle.run_tests()
    test_environment.run_tests()
    test_sensors.run_tests()
    test_vehicle_placement.run_tests()
    test_simulation.run_tests()
    initialization_tests()
    test_graphics()
    print("App Test: All tests PASSED.")


if __name__ == "__main__":
    # initialization_tests()
    run_all_tests()
