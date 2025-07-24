import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from point import Point
from vehicle import Vehicle
import math

from sensor_array import SensorArray
import graphics

import layout_utils as LU
from environment import Environment


class VehicleEditorApp(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.root = self

        plt.rcParams["font.family"] = "Consolas"

        # Create a Matplotlib figure and axes
        self.fig, self.ax = plt.subplots()

        self.v = Vehicle()
        self.v.vehicle_setup(Point(0, 0), 0, 0)
        self.num_sensors = 5
        self.sensor_length = 100
        self.angle_spread = math.pi
        self.sensor_array = SensorArray(
            self.num_sensors, self.sensor_length, self.angle_spread
        )
        self.min_angle = math.pi / 8
        self.max_angle = 2 * math.pi
        self.angle_offset = 0.0

        self.env = None
        # self.step = 5
        # self.angle = sensor_angle_range_deg  # Default starting angle

        self.__define_buttons()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # Add this line to bind the close protocol
        # self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # self.draw_vehicle()

    def __define_buttons(self):

        modify_frame = tk.Frame(self)
        modify_frame.pack(side=tk.RIGHT, expand=True, padx=10, pady=10)
        spin_width = 15

        r = 0
        select_layout_button = ttk.Button(
            modify_frame, text="Select Layout", command=self.select_layout
        )
        select_layout_button.grid(row=r, columnspan=2, pady=10)
        r += 1

        longitude_label = tk.Label(modify_frame, text="Longitude:")
        longitude_label.grid(row=r, column=0, pady=0)

        # Longitude slide bar
        self.longitude_slider = tk.Scale(
            modify_frame,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            command=self.update_longitude,
        )
        self.longitude_slider.set(0.5)  # Set initial value to 0.5
        self.longitude_slider.grid(row=r, column=1, pady=0)

        r += 1

        latitude_label = tk.Label(modify_frame, text="Latitude:")
        latitude_label.grid(row=r, column=0, pady=0)
        self.latitude_slider = tk.Scale(
            modify_frame,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            command=self.update_latitude,
        )
        self.latitude_slider.set(0.5)  # Set initial value to 0.5
        self.latitude_slider.grid(row=r, column=1, pady=0)

        r += 1

        angle_spread_label = tk.Label(modify_frame, text="Sensor Spread (rad):")
        angle_spread_label.grid(row=r, column=0, pady=0)

        self.angle_spread_slider = tk.Scale(
            modify_frame,
            from_=self.min_angle,
            to=self.max_angle,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            command=self.update_angle_spread,
        )
        self.angle_spread_slider.set(self.sensor_array.sensor_angle_spread)
        self.angle_spread_slider.grid(row=r, column=1, pady=0)
        r += 1

        angle_offset_label = tk.Label(modify_frame, text="Angle Offset (rad):")
        angle_offset_label.grid(row=r, column=0, pady=0)

        self.angle_offset_slider = tk.Scale(
            modify_frame,
            from_=-math.pi,
            to=math.pi,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            command=self.update_angle_offset,
        )
        self.angle_offset_slider.set(self.angle_offset)
        self.angle_offset_slider.grid(row=r, column=1, pady=0)

        r += 1

        num_sensor_label = tk.Label(modify_frame, text="Number of Sensors:").grid(
            row=r, column=0, pady=5
        )
        self.num_sensors_spinbox = ttk.Spinbox(
            modify_frame, from_=1, to=50, width=spin_width
        )
        self.num_sensors_spinbox.grid(row=r, column=1, pady=5)
        self.num_sensors_spinbox.set(self.num_sensors)

        r += 1
        sensor_length_label = tk.Label(modify_frame, text="Sensor Length:").grid(
            row=r, column=0, pady=5
        )
        self.sensor_length_spinbox = ttk.Spinbox(
            modify_frame, from_=1, to=1000, width=spin_width
        )
        self.sensor_length_spinbox.grid(row=r, column=1, pady=5)
        self.sensor_length_spinbox.set(self.sensor_length)

        r += 1
        speed_label = tk.Label(modify_frame, text="Initial Speed:").grid(
            row=r, column=0, pady=5
        )
        self.speed_spinbox = ttk.Spinbox(
            modify_frame,
            from_=5.0,
            to=120.0,
            format="%.3f",
            increment=0.1,
            width=spin_width,
        )
        self.speed_spinbox.grid(row=r, column=1, pady=5)
        formatted_value = f"{self.v.speed_mph:.3f}"  # Limit to 2 decimal places
        self.speed_spinbox.delete(0, tk.END)
        self.speed_spinbox.insert(0, formatted_value)
        # self.speed_spinbox.set()
        r += 1

        update_vehicle_btn = ttk.Button(
            modify_frame, text="Update Vehicle", command=self.update_vehicle
        ).grid(row=r, columnspan=2, pady=5)

        r += 1
        vehicle_info_label = tk.Label(modify_frame, text="Vehicle Info:").grid(
            row=r, columnspan=2, pady=10
        )
        self.info_text = tk.StringVar(value=self.get_info_text())
        self.vehicle_info = tk.Label(modify_frame, textvariable=self.info_text).grid(
            row=r, columnspan=2, pady=5
        )

        r += 1
        save_vehicle_info_btn = ttk.Button(
            modify_frame,
            text="Save Vehicle Info",
            command=self.save_vehicle_info_to_file,
        )
        save_vehicle_info_btn.grid(row=r, column=0, pady=10)
        load_vehicle_info_btn = ttk.Button(
            modify_frame,
            text="Load Vehicle Info",
            command=self.load_vehicle_info_from_file,
        )
        load_vehicle_info_btn.grid(row=r, column=1, pady=10)

    def update_vehicle(self):
        if self.env is None:
            self.info_text.set("No environment selected.")
            return
        self.num_sensors = int(self.num_sensors_spinbox.get())
        self.sensor_length = float(self.sensor_length_spinbox.get())
        self.sensor_array = SensorArray(
            self.num_sensors, self.sensor_length, self.angle_spread
        )
        self.v.speed_fps = self.v.mph_to_fps(float(self.speed_spinbox.get()))
        self.draw_vehicle()

    def select_layout(self):
        ctrl_pts, lane_width, closed_loop = LU.load_lane_from_file()
        if ctrl_pts is not None:
            lane = LU.Lane(ctrl_pts, lane_width, closed_loop)
            self.env = Environment(lane)
            self.update_placement()
            self.draw_vehicle()

    def update_placement(self):
        if self.env is None:
            self.info_text.set("No environment selected.")
            return

        # Update the vehicle's position based on the longitude and latitude values
        center, heading = self.env.position_from_coordinates(
            self.longitude, self.latitude, self.angle_offset
        )
        self.v.vehicle_setup(center, heading, float(self.speed_spinbox.get()))
        self.sensor_array.sensor_angle_spread = self.angle_spread
        self.sensor_array.setup_sensors(self.sensor_length)
        self.sensor_array.update_sensors(self.v.center_point, self.v.heading)
        self.draw_vehicle()

    def update_longitude(self, value):
        self.longitude = float(value)
        self.update_placement()

    def update_latitude(self, value):
        self.latitude = float(value)
        self.update_placement()

    def update_angle_spread(self, value):
        self.angle_spread = float(value)
        self.update_placement()

    def update_angle_offset(self, value):
        self.angle_offset = float(value)
        self.update_placement()

    def get_info_text(self):
        s = (
            f"Number of sensors: {self.num_sensors}"
            f"\nSensor length (ft): {self.sensor_length}"
            f"\nSensor angle range (rad): {self.angle_spread}"
            f"\nSpeed (mph): {self.v.speed_mph: .3f}"
        )
        return s

    def update_info_text(self):
        # self.vehicle_info.config(text=self.info_text)
        self.info_text.set(self.get_info_text())

    def draw_vehicle(self):
        if self.env is None:
            self.info_text.set("No environment selected.")
            return
        self.update_info_text()
        self.ax.clear()
        graphics.plot_environment(self.env, self.ax)
        graphics.plot_vehicle(self.v, self.ax)
        self.sensor_array.update_sensors(self.v.center_point, self.v.heading)
        graphics.plot_sensors(self.sensor_array, self.ax)
        self.canvas.draw()

    def save_vehicle_info_to_file(self):
        if self.env is None:
            self.info_text.set("No environment selected.")
            return

        presets = {
            "longitude": self.longitude,
            "latitude": self.latitude,
            "angle_offset": self.angle_offset,
            "speed": self.v.speed_mph,
            "num_sensors": self.num_sensors,
            "sensor_length": self.sensor_array.sensors[0].sensor_length,
            "sensor_angle_spread": self.sensor_array.sensor_angle_spread,
        }
        # save_vehicle_presets(presets)
        LU.save_vehicle_setup(None, presets)

    def load_vehicle_info_from_file(self):
        if self.env is None:
            self.info_text.set("No environment selected.")
            return

        data = LU.load_vehicle_setup_from_file()
        if data is None:
            self.info_text.set("Failed to load vehicle data.")
            return

        self.longitude = float(data["longitude"])
        self.latitude = float(data["latitude"])
        self.angle_offset = float(data["angle_offset"])
        speed = float(data["speed"])
        self.num_sensors = int(data["num_sensors"])
        self.sensor_length = float(data["sensor_length"])
        self.angle_spread = float(data["sensor_angle_spread"])

        self.set_spinboxes(
            self.num_sensors,
            self.sensor_length,
            speed,
        )

        self.set_sliders(self.longitude, self.latitude, self.angle_offset)
        self.sensor_array = SensorArray(
            self.num_sensors, self.sensor_length, self.angle_spread
        )
        self.update_placement()
        self.update_vehicle()
        self.draw_vehicle()

    def set_spinboxes(self, num_sensors, sensor_len, speed):
        self.num_sensors_spinbox.set(num_sensors)
        self.sensor_length_spinbox.set(sensor_len)
        formatted_value = f"{speed:.2f}"  # Limit to 2 decimal places
        self.speed_spinbox.delete(0, tk.END)
        self.speed_spinbox.insert(0, formatted_value)
        # self.speed_spinbox.set(speed)

    def set_sliders(self, longitude, latitude, angle_offset):
        self.longitude_slider.set(longitude)
        self.latitude_slider.set(latitude)
        self.angle_offset_slider.set(angle_offset)

    # def on_close(self):
    #     """Handle the close event to stop the program."""
    #     self.quit()  # Stop the Tkinter main loop
    #     self.destroy()  # Destroy the window and release resources
    #     sys.exit()  # Exit the Python interpreter


if __name__ == "__main__":
    root = tk.Tk()
    app = VehicleEditorApp(root)
    app.pack(expand=True, fill=tk.BOTH)
    root.mainloop()
