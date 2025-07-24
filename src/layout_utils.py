from point import Point
from lane import Lane
from tkinter import filedialog
from vehicle import Vehicle
from sensor_array import SensorArray


def load_lane_from_file(file_path: str = None) -> tuple[list[Point], float, bool]:
    """Loads a lane from a file. The file should contain the control points, lane width and closed loop parameter.
    - The closed loop parameter is expected to be in the format: `L:x` where x is either True or False.
    - The lane width is expected to be a float value in the format: `W:x` where x is the width of the lane.
    - The control points are expected to be in the format: `P:x,y` where x and y are the coordinates of the control points.
    The function returns a tuple containing the control points, lane width and closed loop parameter.

    If the file_path is None, a file dialog will be opened to select the file.

    Args:
        file_path (str): Path to the file containing the lane data.
    Returns:
        tuple([list[Point], float, bool]): Tuple containing the control points, lane width and closed loop parameter.
    """
    if file_path is None:
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

    if file_path is None or len(file_path) == 0:
        return None

    with open(file_path, "r") as file:
        lines = file.readlines()
        control_points = []
        lane_width = 0.0
        closed_loop = False
        for line in lines:
            if line.startswith("L:"):
                closed_loop = line[2:].strip().lower() == "true"
            elif line.startswith("W:"):
                lane_width = float(line[2:].strip())
            elif line.startswith("P:"):
                point_data = line[2:].strip().split(",")
                point = Point(float(point_data[0]), float(point_data[1]))
                control_points.append(point)

    return control_points, lane_width, closed_loop


def save_lane_to_file(filename: str, lane: Lane) -> None:
    """Saves the lane to a file. The file will contain the control points, lane width and closed loop parameter.
    - The closed loop parameter is expected to be in the format: `L:x` where x is either True or False.
    - The lane width is expected to be a float value in the format: `W:x` where x is the width of the lane.
    - The control points are expected to be in the format: `P:x,y` where x and y are the coordinates of the control points.
    Args:
        filename (str): Path to the file where the lane data will be saved.
        lane (Lane): Lane object to be saved.
    """
    save_layout_to_file(
        filename=filename,
        ctrl_pts=lane.control_points,
        closed_loop=lane.closed_loop,
        lane_width=lane.width,
    )
    # with open(filename, "w") as file:
    #     file.write(f"L:{lane.closed_loop}\n")
    #     file.write(f"W:{lane.width}\n")
    #     for point in lane.control_points:
    #         file.write(f"P:{point.x},{point.y}\n")


def save_layout_to_file(
    filename: str, ctrl_pts: list, closed_loop: bool, lane_width: float
) -> None:
    """
    Saves the lane to a file. The file will contain the control points, lane width and closed loop parameter.
    - The closed loop parameter is expected to be in the format: `L:x` where x is either True or False.
    - The lane width is expected to be a float value in the format: `W:x` where x is the width of the lane.
    - The control points are expected to be in the format: `P:x,y` where x and y are the coordinates of the control points.

    Args:
        filename (str): Path to the file where the lane data will be saved.
        ctrl_pts (list): List of control points to be saved.
        closed_loop (bool): Whether the lane is a closed loop or not.
        lane_width (float): Width of the lane.
    """
    if filename is None:
        filename = filedialog.asksaveasfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

    if filename is None or len(filename) == 0:
        return None

    if not filename.endswith(".txt"):
        filename = filename + ".txt"

    with open(filename, "w") as file:
        file.write(f"L:{closed_loop}\n")
        file.write(f"W:{lane_width}\n")
        for point in ctrl_pts:
            file.write(f"P:{point.x},{point.y}\n")


def save_vehicle_setup(filename: str, save_data: str):
    """Saves the vehicle setup to a file. The file will contain the vehicle's longitude, latitude, speed, and sensor array parameters.
    - The vehicle's longitude and latitude are expected to be in the format: `V:x,y` where x is the longitude and y is the latitude.
    - The sensor angle offset is expected to be in the format: `O:x` where x is the angle offset of the sensors.
    - The number of sensors is expected to be in the format: `N:x` where x is the number of sensors.
    - The sensor length is expected to be in the format: `L:x` where x is the length of the sensors.
    - The sensor angle spread is expected to be in the format: `A:x` where x is the angle spread of the sensors.
    - The vehicle's speed is expected to be in the format: `S:x` where x is the speed of the vehicle.


    Args:
        filename (str): _description_
        layout_filename (str): _description_
        longitude (float): _description_
        latitude (float): _description_
        speed (float): _description_
        sensor_array (SensorArray): _description_

    Returns:
        _type_: _description_
    """
    if filename is None:
        filename = filedialog.asksaveasfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

    if filename is None or len(filename) == 0:
        return None

    if not filename.endswith(".txt"):
        filename = filename + ".txt"

    with open(filename, "w") as file:
        for k, v in save_data.items():
            file.write(f"{k}:{v}\n")


def load_vehicle_setup_from_file(file_path: str = None) -> dict:
    """Loads the vehicle setup from a file. The file should contain the vehicle's longitude, latitude, speed, and sensor array parameters of number of sensors,
    length of sensors, and sensor angle spread.

    If the file_path is None, a file dialog will be opened to select the file.

    Args:
        file_path (str): Path to the file containing the vehicle setup data.

    Returns:
        dict: Dictionary containing the vehicle and sensor array parameters.
        longitude, latitude, speed, and sensor array parameters of number of sensors, length of sensors, and sensor angle spread.
    """
    if file_path is None:
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

    if file_path is None or len(file_path) == 0:
        return None

    with open(file_path, "r") as file:
        lines = file.readlines()
        presets = {}
        for line in lines:
            data = line.split(":")
            if len(data) != 2:
                continue
            key = data[0].strip()
            value = data[1].strip()
            presets[key] = value

    return presets
