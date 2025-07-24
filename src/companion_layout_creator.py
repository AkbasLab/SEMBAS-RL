import tkinter as tk
from tkinter import scrolledtext, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from lane import Lane
from point import Point
import layout_utils as LU
from vehicle import Vehicle
import sys
import graphics


class LayoutCreatorApp(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.root = self

        plt.rcParams["font.family"] = "Consolas"

        # Create a Matplotlib figure and axes
        self.figure, self.ax = plt.subplots()

        self.legend_visible = False  # Initialize legend visibility
        self.closed_loop_lane = False
        self.lane_width = 12.0  # Default lane width
        self.last_selected = None
        self.plot_center = False

        self.__initialize_attributes()

        self.__create_frames()
        self.__create_canvas()
        self.__create_buttons()

        # Bind events
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        # Bind the close event
        # self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        # self.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.draw()

    def __initialize_attributes(self):
        self.points = []  # Start with an empty list of points

        self.selected_point = None
        self.done = False
        self.mode = None
        self.lane = None

    # ---------------- Setup Methods ---------------#
    def __create_frames(self):
        # Create a frame for the Matplotlib plot
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a frame for the log area
        self.log_frame = tk.Frame(self.root)
        self.log_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        instructions = (
            "Use the buttons below to add, remove, or edit lane control points. "
            "Set the lane width using the input box. "
            "Use the checkboxes to customize the how the lane is displayed."
        )

        # Create a frame for the instructions area
        instructions_label = tk.Label(self.log_frame, text=instructions, wraplength=400)
        instructions_label.pack(side=tk.TOP, pady=(0, 10))

        self.button_frame = tk.Frame(self.log_frame)
        self.button_frame.pack(side=tk.BOTTOM, pady=5)

    def __create_canvas(self):
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Create the log area
        self.log_area = scrolledtext.ScrolledText(self.log_frame, width=40, height=20)
        self.log_area.pack()
        self.log_area.tag_config("red", foreground="red")

    def __create_buttons(self):
        w = 20
        row = 0
        add_ctrl_pts = ttk.Button(
            self.button_frame,
            text="Add Control Points",
            command=self.add_control_points,
            width=18,
        )
        add_ctrl_pts.grid(row=row, columnspan=2, pady=8, padx=8)
        row += 1

        edit_button = ttk.Button(
            self.button_frame, text="Select Mode", command=self.edit_button, width=18
        )
        edit_button.grid(row=row, column=0, pady=8, padx=8)

        remove_button = ttk.Button(
            self.button_frame, text="Remove Point", command=self.remove_key, width=18
        )
        remove_button.grid(row=row, column=1, pady=8, padx=8)
        row += 1

        lane_width_label = tk.Label(self.button_frame, text="Lane Width (ft):")
        lane_width_label.grid(row=row, column=0, pady=5)

        self.lane_width_spinbox = ttk.Spinbox(
            self.button_frame, from_=2, to=20, width=16
        )
        self.lane_width_spinbox.grid(row=row, column=1, pady=5)
        self.lane_width_spinbox.set(self.lane_width)
        row += 1

        self.legend = tk.BooleanVar()
        legend_check = tk.Checkbutton(
            self.button_frame,
            text="Legend On       ",
            variable=self.legend,
            command=self.toggle_legend,
            width=w,
            justify="left",
        )
        legend_check.grid(row=row, column=0, sticky="w", pady=8)

        self.loop = tk.BooleanVar()
        closed_loop_check = tk.Checkbutton(
            self.button_frame,
            text="Closed Loop     ",
            variable=self.loop,
            command=self.toggle_closed_loop,
            width=w,
            justify="left",
        )
        closed_loop_check.grid(row=row, column=1, sticky="w", pady=8)
        row += 1

        self.center = tk.BooleanVar()
        plot_ctrl_pts_check = tk.Checkbutton(
            self.button_frame,
            text="Plot Control Points",
            variable=self.center,
            command=self.toggle_control_points,
            width=w,
        )
        plot_ctrl_pts_check.grid(row=row, columnspan=2, padx=8, pady=8)
        row += 1

        plot_button = ttk.Button(
            self.button_frame, text="Plot Lane", command=self.plot_lane, width=18
        )
        plot_button.grid(row=row, columnspan=2, pady=8)
        row += 1

        save_button = ttk.Button(
            self.button_frame, text="Save Layout", command=self.save_points, width=18
        )
        save_button.grid(row=row, column=0, pady=8, padx=8)

        load_button = ttk.Button(
            self.button_frame, text="Load Layout", command=self.load_points, width=18
        )
        load_button.grid(row=row, column=1, pady=8, padx=8)
        row += 1

        reset_button = ttk.Button(
            self.button_frame,
            text="Reset Canvas/Log",
            command=self.reset_canvas,
            width=18,
        )
        reset_button.grid(row=row, column=0, pady=8, padx=8)

        clear_log_button = ttk.Button(
            self.button_frame, text="Clear Log", command=self.clear_log, width=18
        )
        clear_log_button.grid(row=row, column=1, pady=8, padx=8)

        # Configure column weights to ensure consistent widths
        self.button_frame.grid_columnconfigure(0, weight=1, uniform="col")
        self.button_frame.grid_columnconfigure(1, weight=1, uniform="col")

    # ------------ End of Setup Methods ------------#

    # ---------------- Draw Methods ---------------#
    def draw(self):
        self.ax.clear()

        self.draw_points(self.points, "ro-", "Control Points")

        if self.last_selected is not None:
            self.ax.plot(
                self.points[self.last_selected][0],
                self.points[self.last_selected][1],
                "bo",
                label="Selected Point",
            )

        plt.tight_layout()
        self.ax.set_xlim(0, 400)
        self.ax.set_ylim(0, 400)

        self.toggle_legend()

        self.canvas.draw()

    def draw_points(self, points, color, label):
        # Plot the line connecting the points
        if points:
            xs, ys = zip(*points)
            self.ax.plot(xs, ys, color, label=label)

            # Annotate each point with its index
            for i, (x, y) in enumerate(points):
                self.ax.text(x, y, str(i), fontsize=12, ha="right")  # Index label

    # ------------ End of Draw Methods ------------#

    # ---------------- Point Selection Methods ---------------#
    def drag_point(self, event):
        self.last_selected = None
        # Check if the user is clicking on an existing point to drag or remove
        if self.select_point(event, self.points):
            self.log_area.insert(
                tk.END, f"Control point {self.selected_point} selected.\n"
            )
            return

    def select_point(self, event, pt_list):
        selection_threshold = 3
        for i, (x, y) in enumerate(pt_list):
            if (
                abs(x - event.xdata) < selection_threshold
                and abs(y - event.ydata) < selection_threshold
            ):
                self.selected_point = i
                return True
        return False

    # ------------ End of Point Selection Methods ------------#

    # --------------- Mouse Event Method ---------------#
    def on_press(self, event):
        if event.inaxes != self.ax or self.done:
            return
        if self.mode == "add":
            # Add a new point at the mouse click location
            self.points.append((event.xdata, event.ydata))
            self.draw()
        else:
            self.drag_point(event)

    def on_motion(self, event):
        if self.selected_point is None or self.done or self.mode != "edit":
            return
        if event.inaxes != self.ax:
            return
        self.points[self.selected_point] = (event.xdata, event.ydata)
        self.draw()

    def on_release(self, event):
        self.last_selected = self.selected_point
        self.selected_point = None  # Reset the selected point on release
        self.draw()

    # ------------ End of Mouse Event Method ------------#

    # ------------Button Methods------------ #
    def edit_button(self):
        # Check if lane is displayed and clear it
        if self.lane is not None:
            self.ax.clear()  # Clear the lane display
            self.lane = None
            self.draw()  # Redraw the plot with points only
        self.mode = "edit"  # Switch to edit mode
        self.log_area.insert(tk.END, "Edit mode.\n")

    def remove_key(self):
        # Check if there is a point selected
        if self.last_selected == None:
            self.log_area.insert(tk.END, "You have not selected a point.\n")
            return
        self.points.pop(self.last_selected)
        self.log_area.insert(tk.END, "Removed a control point.\n")
        self.selected_point = None
        self.last_selected = None
        self.draw()

    def add_control_points(self):
        self.mode = "add"
        self.log_area.insert(tk.END, "Adding a new control point.\n")

    def save_points(self):
        if self.points == []:
            self.log_area.insert(tk.END, "No points to save.\n")
            return
        points = [Point(x, y) for x, y in self.points]
        self.lane_width = float(self.lane_width_spinbox.get())
        LU.save_layout_to_file(None, points, self.closed_loop_lane, self.lane_width)
        self.log_area.insert(tk.END, f"Layout saved.\n")

    def load_points(self):
        layout = LU.load_lane_from_file()
        if layout == None:
            self.log_area.insert(tk.END, f"No file selected.\n")
            return
        control_points, lane_width, closed_loop = layout
        self.lane_width = lane_width
        self.lane_width_spinbox.set(self.lane_width)
        self.closed_loop_lane = closed_loop
        self.loop.set(self.closed_loop_lane)
        self.points = []
        for pt in control_points:
            self.points.append((pt.x, pt.y))

        self.draw()  # Redraw the plot with loaded points
        self.log_area.insert(tk.END, f"Loaded layout.\n")

    def toggle_legend(self):
        self.legend_visible = self.legend.get()
        if self.legend_visible:
            self.ax.legend()
        elif self.ax.get_legend() is not None:
            self.ax.get_legend().remove()
        self.canvas.draw_idle()  # Update only the legend without redrawing the entire plot

    def toggle_closed_loop(self):
        self.closed_loop_lane = self.loop.get()
        l = "off"
        if self.closed_loop_lane:
            l = "on"
        self.log_area.insert(tk.END, f"Closed loop lane turned {l}.\n")

    def toggle_control_points(self):
        self.plot_center = self.center.get()

    def plot_lane(self):
        if len(self.points) < 2:
            self.log_area.insert(tk.END, "Not enough points to plot a lane.\n")
            return
        self.ax.clear()
        control_points = [Point(x, y) for x, y in self.points]
        self.lane_width = float(self.lane_width_spinbox.get())
        self.lane = Lane(
            control_points=control_points,
            lane_width=self.lane_width,
            closed_loop=self.closed_loop_lane,
        )
        graphics.plot_lane(lane=self.lane, ax=self.ax)
        if self.plot_center:
            x = [x for x, _ in self.points]
            y = [y for _, y in self.points]
            self.ax.plot(x, y, "g-o", label="Control Points")

        self.canvas.draw()
        self.log_area.insert(tk.END, "Displayed lane from points.\n")

    # def plot_control_points(self)

    def reset_canvas(self):
        self.ax.clear()
        self.__initialize_attributes()
        self.draw()
        self.clear_log()
        self.log_area.insert(tk.END, f"Canvas reset.\n")

    def clear_log(self):
        self.log_area.delete(1.0, tk.END)

    def on_close(self):
        print("Window closed. Exiting script...")
        self.root.destroy()
        sys.exit()


# ------------End of button methods------------ #

if __name__ == "__main__":
    root = tk.Tk()
    app = LayoutCreatorApp(root)
    root.mainloop()
