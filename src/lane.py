from scipy.interpolate import CubicSpline
import numpy as np


class Lane:
    def __init__(
        self,
        control_points: list[np.ndarray],
        lane_width: float = 12.0,
        closed_loop: bool = False,
    ):
        """Represents a lane in an environment. Based on given control points, it creates a lane with a given width.
        The lane is represented as a series of splines through the control points. The lane is created based on the closed_loop parameter.

        Args:
            control_points (list[Point]): List of Point object with x, y values to create splines through.
            lane_width (float): Width (in feet) of the lane. Defaults to 12.0
            closed_loop (bool, optional): If True, lane is calculated so that the end of the lane connects with the start. Defaults to False.
        """
        self.lane_width = lane_width
        self.closed_loop = closed_loop
        self.lane_setup(control_points, lane_width, closed_loop)

    def lane_setup(
        self,
        control_points: list[np.ndarray],
        lane_width: float = None,
        closed_loop: bool = None,
    ):
        """Sets up the lane based on the given control points, lane width and closed loop parameter.

        Args:
            control_points (list[Point]): List of Point object with x, y values to create spline through.
            lane_width (float, optional): Width (in feet) of the lane. Defaults to None.
            closed_loop (bool, optional): If True, lane is calculated so that the end of the lane connects with the start. Defaults to None.
        """
        self.control_points = np.vstack(control_points)
        if lane_width is not None:
            self.lane_width = lane_width
        if closed_loop is not None:
            self.closed_loop = closed_loop

        self.center_line = self.calculate_center(self.control_points)
        self.left_edge, self.right_edge = self.calculate_edges(self.center_line)

    @staticmethod
    def _distance_between(p1: np.ndarray, p2: np.ndarray) -> float:
        return np.linalg.norm(p2 - p1)

    def nearest_neighbor(self, p: np.ndarray) -> tuple[int, np.ndarray]:
        "Returns the index of the point closest to @p. O(n) search time"
        nearest = (
            self._distance_between(p, self.control_points[0]),
            0,
        )
        for i, lp in enumerate(self.control_points[1:]):
            d = self._distance_between(p, lp)
            if d < nearest[0]:
                nearest = (d, i + 1)

        return nearest[1]

    def calculate_center(
        self, control_points: np.ndarray
    ) -> tuple[list[np.ndarray], list[float], list[float]]:
        """Calculates the center line of the lane based on the given control points.

        Args:
            control_points (list[Point]): List of Point object with x, y values to create spline through.

        Returns:
            tuple(list[Point], list[float], list[float]):
            - list[Point]: List of Point object representing the center line of the lane.
            - list[float], list[float]: Lists representing the x and y center values
        """
        num_points = 500  # Number of points to sample along the spline

        self.spline_x, self.spline_y, self.t = self.__calculate_xy_spline(
            control_points
        )

        # Generate points along the spline path
        self.t_center = np.linspace(self.t.min(), self.t.max(), num_points)
        x_center = self.spline_x(self.t_center)
        y_center = self.spline_y(self.t_center)
        centers = np.array([x_center, y_center]).T

        if self.closed_loop:
            # Appends the reference to the first point rather than creating a new point at the exact same spot
            centers = np.vstack([centers, centers[0]])

        return centers

    def __calculate_xy_spline(self, control_points: np.ndarray):
        """Calculates the x and y splines from the given x and y point values.

        Parameters:
            x_pts (list[float]): list of x values
            y_pts (list[float]): list of y values

        Returns:
            spline_x (CubicSpline): cubic spline x values.
            spline_y (CubicSpline): cubic spline y values.
            t (list[float]): list of values used as x values for CubicSpline creation.
        """
        # Calculate cumulative distances along the control points for parameterization
        distances = np.linalg.norm(np.diff(control_points.T).T, axis=1)
        # distances = np.sqrt(np.diff(x_pts) ** 2 + np.diff(y_pts) ** 2)
        t = np.concatenate(
            ([0], np.cumsum(distances))
        )  # Parameter based on cumulative arc length

        # If closed, make sure the loop closes back on itself
        if self.closed_loop:
            control_points = np.vstack([control_points, control_points[0]])
            t = np.append(t, t[-1] + distances[0])  # Extend t to keep periodicity

        # Create cubic splines with periodic boundary if closed
        bc_type = "periodic" if self.closed_loop else "not-a-knot"
        spline_x = CubicSpline(t, control_points.T[0], bc_type=bc_type)
        spline_y = CubicSpline(t, control_points.T[1], bc_type=bc_type)

        return spline_x, spline_y, t

    def __calculate_slope_vectors(self):
        """Calculates the slope vectors to offset from the center line to created the lane edges.

        Returns:
            tuple:
            - slope_vector_x (float): slope vector value in the x direction
            - slope_vector_y (float): slope vector value in the y direction
        """
        # Calculate derivatives for x and y to get slope vectors
        dx_dt = self.spline_x.derivative()(self.t_center)
        dy_dt = self.spline_y.derivative()(self.t_center)

        # Magnitude of the derivative vector
        magnitude = np.sqrt(dx_dt**2 + dy_dt**2)
        scale_factor = (self.lane_width / 2) / magnitude

        # Scaled slope vector components, rotated for perpendicular lane edges
        slope_vector_x = -dy_dt * scale_factor  # Rotate by 90 degrees
        slope_vector_y = dx_dt * scale_factor  # Rotate by 90 degrees

        return np.array([slope_vector_x, slope_vector_y]).T

    def calculate_edges(self, center_pts: np.ndarray):
        """Calculates the left and right edges of the lane based on the center line and lane width.
        Edges are calculated by offsetting the center line points by half the lane width in the perpendicular direction.
        The perpendicular direction is determined by the tangent of the spline at each point.
        The left edge is offset to the left of the center line and the right edge is offset to the right.

        Returns:
            tuple[list[Point], list[Point]]: Tuple of two lists of Point object representing the left and right edges of the lane.
        """

        slope_vectors = self.__calculate_slope_vectors()

        # # Calculate the right edge coordinates
        right_edge = self.calculate_edge_coordinates(
            center_pts=(center_pts),
            slop_vectors=slope_vectors,
            multiplier=-1,
        )

        # Calculate the left edge coordinates
        left_edge = self.calculate_edge_coordinates(
            center_pts=(center_pts),
            slop_vectors=slope_vectors,
            multiplier=1,
        )

        # If closed, ensure the edges connect at both ends
        if self.closed_loop:
            # Append the first point to the end of the edges
            right_edge = np.vstack((right_edge, right_edge[0]))
            left_edge = np.vstack((left_edge, left_edge[0]))

        return left_edge, right_edge

    def calculate_edge_coordinates(
        self, center_pts: np.ndarray, slop_vectors: np.ndarray, multiplier: float
    ):
        """Calculates the coordinates of the lane at a given parameter t.

        Args:
            t (float): Parameter value to calculate the coordinates at.

        Returns:
            tuple[float, float]: Tuple of x and y coordinates at the given parameter t.
        """

        if self.closed_loop:
            edge = center_pts[:-1] + slop_vectors * multiplier
        else:
            edge = center_pts + slop_vectors * multiplier
        return edge
