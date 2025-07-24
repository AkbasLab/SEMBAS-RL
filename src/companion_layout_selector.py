import tkinter as tk
from tkinter import ttk, filedialog
import os
import layout_utils as LU
import sys
from lane import Lane
import graphics

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class LayoutSelectorApp(tk.Frame):
    def __init__(self, parent, folder_path, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # GUI Components
        self.file_list = (
            self.get_txt_files(folder_path) if folder_path is not None else []
        )
        self.selected_file = tk.StringVar(
            value=self.file_list[0] if self.file_list else ""
        )
        self.selected_files = []

        other_frame = tk.Frame(self)
        other_frame.pack(side=tk.RIGHT, expand=True, padx=10, pady=10)

        directory_frame = tk.Frame(other_frame)
        directory_frame.pack(side=tk.TOP, expand=True, padx=10, pady=10)
        self.directory_label = tk.Label(
            directory_frame, text="Current Directory: " + str(folder_path), fg="red"
        )
        self.directory_label.pack(side=tk.LEFT, padx=5)
        directory_button = ttk.Button(
            directory_frame, text="Change Directory", command=self.change_directory
        )
        directory_button.pack(side=tk.RIGHT, padx=5)
        self.folder_path = folder_path

        self.list_frame = tk.Frame(other_frame)
        self.list_frame.pack(side=tk.RIGHT, expand=True, padx=10, pady=10)

        # Listbox (file selection)
        ttk.Label(self.list_frame, text="Layout Files:").grid(row=0, column=0, pady=5)
        ttk.Label(self.list_frame, text="Selected Files:").grid(row=0, column=2, pady=5)

        self.select_list = tk.Listbox(
            self.list_frame, height=30, width=25, exportselection=False
        )
        self.select_list.grid(rowspan=2, column=0, padx=2, sticky="w")

        # Buttons to interact between lists
        self.add_button = ttk.Button(
            self.list_frame, text="Add >>", command=self.add_to_selected
        )
        self.add_button.grid(row=1, column=1, pady=5, padx=2)

        self.remove_button = ttk.Button(
            self.list_frame, text="<< Remove", command=self.remove_from_selected
        )
        self.remove_button.grid(row=2, column=1, pady=5, padx=2)

        self.save_list = tk.Listbox(
            self.list_frame,
            height=30,
            width=25,
            selectmode=tk.MULTIPLE,
            exportselection=False,
        )
        self.save_list.grid(row=1, rowspan=2, column=2, padx=2, sticky="e")

        # Button to load and display layout
        ttk.Button(
            self.list_frame, text="View Layout", command=self.display_layout, width=20
        ).grid(row=3, column=0, pady=5, padx=8)
        # Button to save selected files
        ttk.Button(
            self.list_frame,
            text="Save Files",
            command=self.save_selected_files,
            width=20,
        ).grid(row=3, column=2, pady=5, padx=8)

        # Matplotlib figure setup
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(expand=True, fill=tk.BOTH)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        plt.tight_layout()

    def change_directory(self):
        new_directory = filedialog.askdirectory()
        if new_directory:
            self.folder_path = new_directory
            self.refresh_files()
            self.directory_label.config(text="Current Directory: " + self.folder_path)

    def get_txt_files(self, folder_path):
        """Gets all txt files in the specified folder."""
        return (
            [f for f in os.listdir(folder_path) if f.endswith(".txt")]
            if folder_path is not None
            else []
        )

    def refresh_files(self):
        self.select_list.delete(0, tk.END)
        self.file_list = self.get_txt_files(self.folder_path)
        # Add files to Listbox
        for file in self.file_list:
            self.select_list.insert(tk.END, file)

    def display_layout(self):
        """Load the selected file and display its layout."""
        selected_index = (
            self.select_list.curselection()
        )  # Get the index of the selected file
        if not selected_index:
            self.plot_error_message("No file selected to view.")
            return

        selected_file = self.file_list[selected_index[0]]  # Get file name from listbox
        file_path = os.path.join(self.folder_path, selected_file)

        self.ax.clear()
        try:
            ctrl_points, lane_width, closed_loop = LU.load_lane_from_file(file_path)
            lane = Lane(
                control_points=ctrl_points,
                lane_width=lane_width,
                closed_loop=closed_loop,
            )
            # plt.sca(self.ax)
            graphics.plot_lane(lane=lane, ax=self.ax)

            # graphics.show_without_pause()

            # self.ax.legend()
            self.canvas.draw()
        except Exception as e:
            self.plot_error_message("Error loading file.\nIncorrect Format.")
            return

    def add_to_selected(self):
        """Add the currently selected file to the second listbox."""
        selected_index = self.select_list.curselection()
        if selected_index:
            file_name = self.file_list[selected_index[0]]
            if file_name not in self.selected_files:
                self.selected_files.append(file_name)
                self.save_list.insert(tk.END, file_name)
        else:
            self.plot_error_message("No files selected.")

    def remove_from_selected(self):
        """Remove the selected file(s) from the second listbox."""
        selected_indices = self.save_list.curselection()
        if not selected_indices:
            self.plot_error_message("No files selected.")
            return
        for index in reversed(selected_indices):  # Reverse to avoid index shifting
            file_name = self.save_list.get(index)
            self.selected_files.remove(file_name)
            self.save_list.delete(index)

    def save_selected_files(self):
        """Save the list of selected files to a text file."""
        if not self.selected_files:
            self.plot_error_message("No files to save.")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text Files", "*.txt")]
        )
        if save_path:
            with open(save_path, "w") as file:
                file.write(f"{self.folder_path}\n")
                for file_name in self.selected_files:
                    file.write(f"{file_name}\n")
            print(f"Selected files saved to {save_path}")
        return self.selected_files

    def plot_error_message(self, msg: str):
        self.ax.clear()
        self.ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=16, color="red")
        self.canvas.draw()

    def on_close(self):
        print("Window closed. Exiting script...")
        self.destroy()
        sys.exit()


# Path where your txt files are stored
# folder_path = "./layouts"  # Change this path as needed

if __name__ == "__main__":
    folder_path = None
    app = LayoutSelectorApp(folder_path)
    app.mainloop()
