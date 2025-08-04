from vehicle import Vehicle
from lane import Lane
from environment import Environment
import graphics as G
import matplotlib.pyplot as plt


def test_position_from_coordinates():
    control_points = [np.ndarray(100, 200), np.ndarray(300, 200)]
    closed_loop = False
    lane = Lane(control_points=control_points, lane_width=12, closed_loop=closed_loop)
    env = Environment(lane=lane)
    longitude = 0.5
    latitude = 0.5
    angle_offset = 0.0
    v = Vehicle()
    center, heading = env.position_from_coordinates(longitude, latitude, angle_offset)
    v.vehicle_setup(center_point=center, abs_heading=heading, speed_mph=25.0)
    assert (
        199.99999 <= v.center_point.x <= 200.000001
    ), f"Vehicle center point X {v.center_point.x} does not match expected value."
    assert (
        199.99999 <= v.center_point.y <= 200.000001
    ), "Vehicle center point Y does not match expected value."
    # assert (
    #     199.99999 <= v.get_heading_point().x <= (200.0000001)
    # ), "Vehicle heading point X does not match expected value."
    # assert (
    #     199.99999 <= v.get_heading_point().y <= 200.0000001
    # ), "Vehicle heading point Y does not match expected value."
    print(
        "Environemnt Test: Vehicle position from coordinates - test PASSED. CONFIRM VISUALIZATION."
    )
    G.plot_environment(env)
    G.plot_vehicle(v)
    plt.title("Vehicle position from coordinates - test")
    plt.xlim(50, 350)
    plt.ylim(150, 250)
    # G.show()


def test_position_in_lane_on_center_line():
    control_points = [np.ndarray(100, 200), np.ndarray(300, 200)]
    closed_loop = False
    lane = Lane(control_points=control_points, lane_width=12, closed_loop=closed_loop)
    env = Environment(lane=lane)
    v = Vehicle()
    v.vehicle_setup(center_point=np.ndarray(200, 200), abs_heading=0.0, speed_mph=25.0)
    in_lane = env.point_in_lane(v.center_point)
    assert in_lane, "Vehicle is not in lane when it should be."
    center_pos, left_edge_pos, right_edge_pos = env.point_position_in_lane(
        v.center_point
    )
    assert (
        -0.000001 <= center_pos <= 0.000001
    ), "Vehicle center position should be directly center of the lane."
    assert (
        left_edge_pos > 0 and right_edge_pos > 0
    ), "Both left and right edges should be positive since the vehicle is on the center line."
    print(
        "Environemnt Test: Vehicle position in lane - on center line - test PASSED. CONFIRM VISUALIZATION."
    )
    G.plot_environment(env)
    G.plot_vehicle(v)
    plt.title("Vehicle position in lane - on center line - test")
    plt.xlim(50, 350)
    plt.ylim(150, 250)


def test_position_in_lane_out_of_bounds():
    control_points = [np.ndarray(100, 200), np.ndarray(300, 200)]
    closed_loop = False
    lane = Lane(control_points=control_points, lane_width=12, closed_loop=closed_loop)
    env = Environment(lane=lane)
    v = Vehicle()
    v.vehicle_setup(center_point=np.ndarray(200, 207), abs_heading=0.0, speed_mph=25.0)
    in_lane = env.point_in_lane(v.center_point)
    assert not in_lane, "Vehicle is not in lane when it should be."
    center_pos, left_edge_pos, right_edge_pos = env.point_position_in_lane(
        v.center_point
    )
    assert (
        -6.99999999 >= center_pos >= -7.000001
    ), "Vehicle center position should be ~ -7ft since lane width is 6.."
    assert (
        left_edge_pos > 0 and right_edge_pos < 0
    ), "Left should be positive and right should be negative since the vehicle is on the center line."
    assert (
        0.999999 <= left_edge_pos <= 1.000001
    ), "Left edge position should be ~1.0 ft from the left edge."
    assert (
        -12.999999 >= right_edge_pos >= -13.000001
    ), "Right edge position should be ~-13ft from the right edge."
    print(
        "Environemnt Test: Vehicle position in lane - out of lane - test PASSED. CONFIRM VISUALIZATION."
    )
    G.plot_environment(env)
    G.plot_vehicle(v)
    plt.title("Vehicle position in lane - out of lane - test")
    plt.xlim(50, 350)
    plt.ylim(150, 250)


def test_position_in_lane_left_of_center():
    control_points = [np.ndarray(100, 200), np.ndarray(300, 200)]
    closed_loop = False
    lane = Lane(control_points=control_points, lane_width=12, closed_loop=closed_loop)
    env = Environment(lane=lane)
    v = Vehicle()
    v.vehicle_setup(center_point=np.ndarray(200, 202), abs_heading=0.0, speed_mph=25.0)
    in_lane = env.point_in_lane(v.center_point)
    assert in_lane, "Vehicle is not in lane when it should be."
    center_pos, left_edge_pos, right_edge_pos = env.point_position_in_lane(
        v.center_point
    )
    assert (
        left_edge_pos > 0 and right_edge_pos < 0
    ), "Left edge should be positive and right edge should be negative since the vehicle is left of the center line."
    assert (
        3.9999999 <= left_edge_pos <= 4.0000001
    ), "Left edge position should be ~4.0 ft from the center line."
    assert (
        -7.9999999 >= right_edge_pos >= -8.0000001
    ), "Right edge position should be ~-8.0 ft from the center line."
    print(
        "Environemnt Test: Vehicle position in lane - left of center line - test PASSED. CONFIRM VISUALIZATION."
    )
    G.plot_environment(env)
    G.plot_vehicle(v)
    plt.title("Vehicle position in lane - left of center line - test")
    plt.xlim(50, 350)
    plt.ylim(150, 250)


def test_position_in_lane_right_of_center():
    control_points = [np.ndarray(100, 200), np.ndarray(300, 200)]
    closed_loop = False
    lane = Lane(control_points=control_points, lane_width=12, closed_loop=closed_loop)
    env = Environment(lane=lane)
    v = Vehicle()
    v.vehicle_setup(center_point=np.ndarray(200, 198), abs_heading=0.0, speed_mph=25.0)
    in_lane = env.point_in_lane(v.center_point)
    assert in_lane, "Vehicle is not in lane when it should be."
    center_pos, left_edge_pos, right_edge_pos = env.point_position_in_lane(
        v.center_point
    )
    assert (
        left_edge_pos < 0 and right_edge_pos > 0
    ), "Left edge should be negative and right edge should be positive since the vehicle is right of the center line."
    assert (
        3.9999999 <= right_edge_pos <= 4.0000001
    ), "Right edge position should be ~4.0 ft from the center line."
    assert (
        -7.9999999 >= left_edge_pos >= -8.0000001
    ), "Left edge position should be ~-8.0 ft from the center line."
    print(
        "Environemnt Test: Vehicle position in lane - right of center line - test PASSED. CONFIRM VISUALIZATION."
    )
    G.plot_environment(env)
    G.plot_vehicle(v)
    plt.title("Vehicle position in lane - right of center line - test")
    plt.xlim(50, 350)
    plt.ylim(150, 250)


def run_tests():
    plt.subplot(2, 3, 1)
    test_position_from_coordinates()
    plt.subplot(2, 3, 2)
    test_position_in_lane_on_center_line()
    plt.subplot(2, 3, 3)
    test_position_in_lane_out_of_bounds()
    plt.subplot(2, 3, 4)
    test_position_in_lane_left_of_center()
    plt.subplot(2, 3, 5)
    test_position_in_lane_right_of_center()
    G.show_without_pause(
        "Vehicle position in lane - right of center line - test", [50, 350], [150, 250]
    )
    print("Environment Test: All tests PASSED.\n")


if __name__ == "__main__":
    run_tests()
