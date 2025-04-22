import matplotlib.pyplot as plt
from lane import Lane
from simulation import Simulation
from vehicle import Vehicle
from environment import Environment
from point import Point
from sensor_array import SensorArray
import numpy as np


def list_points_as_values(point_list: list[Point]) -> tuple[list[float], list[float]]:
    """Takes in a list of points and returns 2 lists: x values and y values.

    Args:
        point_list (list[Point]): List of Point objects.

    Returns:
        tuple(list[float], list[float]): X values and Y values of the given list of points.
    """
    x = []
    y = []
    for p in point_list:
        x.append(p.x)
        y.append(p.y)
    return x, y


def plot_environment(environment: Environment):  # Tested as of 3/29/2025
    """Plots the environment with its lane and vehicle."""
    lane = environment.lane
    plot_lane(lane)  # Plot the lane


def plot_lane(lane: Lane):  # Tested as of 3/29/2025
    """Plots the lane with its center line, left edge and right edge."""
    ax = plt.gca()

    # Plot center line
    center_x, center_y = list_points_as_values(lane.center_line)
    ax.plot(center_x, center_y, "k--", label="Center Line")

    # Plot control points
    ctrl_x, ctrl_y = list_points_as_values(lane.control_points)
    ax.plot(ctrl_x, ctrl_y, "ro", label="Control Points")
    for i, (x, y) in enumerate(zip(ctrl_x, ctrl_y)):
        ax.annotate(text=f"{i}", xy=(x, y), fontsize=8, ha="right")

    # Plot left edge
    left_x, left_y = list_points_as_values(lane.left_edge)
    ax.plot(left_x, left_y, "b-", label="Left Edge")

    # Plot right edge
    right_x, right_y = list_points_as_values(lane.right_edge)
    ax.plot(right_x, right_y, "m-", label="Right Edge")

    ax.fill(
        np.concatenate((left_x, right_x[::-1])),
        np.concatenate((left_y, right_y[::-1])),
        color="gray",
        alpha=0.5,
        label="Lane Area",
    )


def plot_vehicle(vehicle: Vehicle):  # Tested as of 3/29/2025
    """Plots the vehicle with its body, center point and heading."""
    ax = plt.gca()
    # Plot vehicle body
    body_x, body_y = list_points_as_values(vehicle.body.corners)
    ax.fill(body_x, body_y, "b", label="Vehicle Body")

    # Plot vehicle center point
    ax.plot(
        vehicle.center_point.x, vehicle.center_point.y, "ro", label="Vehicle Center"
    )

    # Plot vehicle heading with arrow
    # Getting heading point and scalling by 10.0 so that it is visible.
    d = np.array(vehicle.get_direction_vector()) * 10.0
    heading_point = vehicle.center_point + Point(d[0], d[1])
    x = [vehicle.center_point.x, (heading_point.x)]
    y = [vehicle.center_point.y, (heading_point.y)]
    ax.plot(x, y, "g->", label="Vehicle Heading")
    # ax.annotate(
    #     "",
    #     xy=(x[1], y[1]),
    #     xytext=(x[0], y[0]),
    #     arrowprops=dict(
    #         facecolor="green",
    #         edgecolor="green",
    #         arrowstyle="->",
    #         lw=2,
    #     ),
    # )


def plot_sensors(sensor_array: SensorArray):  # Tested as of 3/29/2025
    """Plots the sensors in the sensor array."""
    ax = plt.gca()
    label = "Sensor"
    for i, sensor in enumerate(sensor_array.sensors):
        ax.plot(
            [sensor.origin_point.x, sensor.end_point.x],
            [sensor.origin_point.y, sensor.end_point.y],
            "r--",
            label=label,
        )
        # ax.annotate(text=f"{i}", xy=(sensor.end_point.values()))
        label = None


def plot_sensor_detections(detection_points, detection_distances):
    """Plots the sensor detections."""
    ax = plt.gca()
    label = "Sensor Detection"
    for i, point in enumerate(detection_points):
        if point is not None:
            ax.plot(point.x, point.y, "kx", label=label)
            ax.annotate(
                text=f"{detection_distances[i]: .2f}",
                xy=(point.x, point.y),
                fontsize=8,
                ha="right",
            )
            label = None


def show(
    title: str = "", x_lim: list[float] = [0, 400], y_lim: list[float] = [0, 400]
):  # Tested as of 3/29/2025
    """Displays the plot."""
    plt.tight_layout()
    plt.legend()
    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])
    plt.title(title)
    plt.pause(0.1)


def show_without_pause(
    title: str = "", x_lim: list[float] = [0, 400], y_lim: list[float] = [0, 400]
):  # Tested as of 3/29/2025
    """Displays the plot."""
    plt.tight_layout()
    plt.legend()
    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])
    plt.title(title)
    plt.show()


def render_simulation(sim: Simulation):
    """Renders the simulation by plotting the evironment, vehicle, sensors, and sensor detections. Clears the current figure first."""
    plt.clf()  # Clear the current figure
    env = sim.environment
    plot_environment(env)  # Plot the environment
    vehicle = sim.vehicle
    plot_vehicle(vehicle)  # Plot the vehicle
    sensors = sim.agent.sensors  # Get the sensor array
    points, detections = sensors.sense(env=env, vehicle=vehicle)  # Update sensor data
    plot_sensors(sensor_array=sensors)
    plot_sensor_detections(
        detection_points=points, detection_distances=detections
    )  # Plot the sensor detections


def render_simulation_subplots(sim: Simulation):
    """Renders the simulation by plotting the evironment, vehicle, sensors, and sensor detections. DOES NOT clear the current figure first."""
    env = sim.environment
    plot_environment(env)  # Plot the environment
    vehicle = sim.vehicle
    plot_vehicle(vehicle)  # Plot the vehicle
    sensors = sim.agent.sensors  # Get the sensor array
    points, detections = sensors.sense(env=env, vehicle=vehicle)  # Update sensor data
    plot_sensors(sensor_array=sensors)
    plot_sensor_detections(
        detection_points=points, detection_distances=detections
    )  # Plot the sensor detections
