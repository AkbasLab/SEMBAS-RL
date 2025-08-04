import numpy as np
from lane import Lane

# from tkinter import filedialog


def load_lane_from_file(file_path: str = None) -> tuple[list[np.ndarray], float, bool]:
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
    # if file_path is None:
    #     file_path = filedialog.askopenfilename(
    #         filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    #     )

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
                point = np.array([float(point_data[0]), float(point_data[1])])
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
    with open(filename, "w") as file:
        file.write(f"L:{lane.closed_loop}\n")
        file.write(f"W:{lane.width}\n")
        for point in lane.control_points:
            file.write(f"P:{point[0]},{point[1]}\n")
