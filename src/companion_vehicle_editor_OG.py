import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from point import Point
from vehicle import Vehicle

from sensor_array import SensorArray


class VehicleEditorApp(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.root = self

        plt.rcParams["font.family"] = "Consolas"

        # Create a Matplotlib figure and axes
        self.fig, self.ax = plt.subplots()

        # self.v = Vehicle()
        # self.sensor_array = SensorArray(self.num_sensors, self.sensor_len)
        # self.min_angle = 30
        # self.max_angle = 180
        # self.step = 5
        # self.angle = sensor_angle_range_deg  # Default starting angle

        # self.__define_buttons()

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
        num_sensor_label = tk.Label(modify_frame, text="Number of Sensors:").grid(
            row=r, column=0, pady=5
        )
        self.num_sensors_spinbox = ttk.Spinbox(
            modify_frame, from_=1, to=50, width=spin_width
        )
        self.num_sensors_spinbox.grid(row=r, column=1, pady=5)
        self.num_sensors_spinbox.set(self.v.num_sensors)

        r += 1
        sensor_length_label = tk.Label(modify_frame, text="Sensor Length:").grid(
            row=r, column=0, pady=5
        )
        self.sensor_length_spinbox = ttk.Spinbox(
            modify_frame, from_=1, to=1000, width=spin_width
        )
        self.sensor_length_spinbox.grid(row=r, column=1, pady=5)
        self.sensor_length_spinbox.set(self.v.sensor_len)

        r += 1
        acceleration_label = tk.Label(modify_frame, text="Initial Acceleration:").grid(
            row=r, column=0, pady=10
        )
        self.acceleration_spinbox = ttk.Spinbox(
            modify_frame, from_=0.1, to=10.0, increment=0.1, width=spin_width
        )
        self.acceleration_spinbox.grid(row=r, column=1, pady=5)
        self.acceleration_spinbox.set(self.v.acceleration)

        r += 1
        speed_label = tk.Label(modify_frame, text="Initial Speed:").grid(
            row=r, column=0, pady=5
        )
        self.speed_spinbox = ttk.Spinbox(
            modify_frame,
            from_=0.1,
            to=100.0,
            format="%.3f",
            increment=0.1,
            width=spin_width,
        )
        self.speed_spinbox.grid(row=r, column=1, pady=5)
        formatted_value = f"{self.v.speed:.3f}"  # Limit to 2 decimal places
        self.speed_spinbox.delete(0, tk.END)
        self.speed_spinbox.insert(0, formatted_value)
        # self.speed_spinbox.set()

        r += 1
        angle_label = tk.Label(modify_frame, text="Sensor Angle Range (°):").grid(
            row=r, column=0, pady=5
        )
        self.angle_spinbox = ttk.Spinbox(
            modify_frame,
            from_=self.min_angle,
            to=self.max_angle,
            increment=self.step,
            width=spin_width,
        )
        self.angle_spinbox.grid(row=r, column=1, pady=5)
        self.angle_spinbox.set(self.v.sensor_angle_range_deg)

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
        self.v.num_sensors = int(self.num_sensors_spinbox.get())
        self.v.sensor_len = float(self.sensor_length_spinbox.get())
        self.v.sensor_angle_range_deg = int(self.angle_spinbox.get())
        self.v.reset_sensors()
        self.v.acceleration = float(self.acceleration_spinbox.get())
        self.v.velocity = self.v.speed_to_velocity(
            self.v.target_point, float(self.speed_spinbox.get())
        )
        self.draw_vehicle()

    def get_info_text(self):
        s = (
            f"Number of sensors: {self.v.num_sensors}"
            f"\nSensor length (ft): {self.v.sensor_len}"
            f"\nSensor angle range (°): {self.v.sensor_angle_range_deg}"
            f"\nAcceleration (mph^2): {self.v.acceleration}"
            f"\nSpeed (mph): {self.v.speed: .3f}"
        )
        return s

    def update_info_text(self):
        # self.vehicle_info.config(text=self.info_text)
        self.info_text.set(self.get_info_text())

    def draw_vehicle(self):
        self.update_info_text()
        self.ax.clear()
        self.v.plot_vehicle_no_detections(self.ax)
        self.canvas.draw()

    def save_vehicle_info_to_file(self):
        presets = {
            "num_sensors": self.v.num_sensors,
            "sensor_len": self.v.sensor_len,
            "acceleration": self.v.acceleration,
            "speed": self.v.speed,
            "sensor_angle_range_deg": self.v.sensor_angle_range_deg,
        }
        # save_vehicle_presets(presets)

    def load_vehicle_info_from_file(self):
        # presets = load_vehicle_presets()
        # self.v.num_sensors = presets["num_sensors"]
        # self.v.sensor_len = presets["sensor_len"]
        # self.v.sensor_angle_range_deg = presets["sensor_angle_range_deg"]
        # self.v.acceleration = presets["acceleration"]
        # self.v.velocity = self.v.speed_to_velocity(self.v.target_point, float(presets["speed"]))
        # self.set_spinboxes(**presets)
        self.update_vehicle()
        self.draw_vehicle()

    def set_spinboxes(self, num_sensors, sensor_len, acceleration, speed, sensor_angle):
        self.num_sensors_spinbox.set(num_sensors)
        self.sensor_length_spinbox.set(sensor_len)
        self.acceleration_spinbox.set(acceleration)
        formatted_value = f"{speed:.2f}"  # Limit to 2 decimal places
        self.speed_spinbox.delete(0, tk.END)
        self.speed_spinbox.insert(0, formatted_value)
        # self.speed_spinbox.set(speed)
        self.angle_spinbox.set(sensor_angle)

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
