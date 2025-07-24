import tkinter as tk
from tkinter import ttk
from companion_layout_selector import LayoutSelectorApp
from companion_layout_creator import LayoutCreatorApp

from companion_vehicle_editor import VehicleEditorApp
from companion_app_help import HelpFrame
import sys


# Main GUI
class CARLOSCompanion(tk.Tk):
    def __init__(self, folder_path="./layouts", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("CARLOS Companion App")
        self.folder_path = folder_path

        # Set global font family and size
        self.configure_global_font(font_family="Consolas", font_size=9)

        # Menu frame at the bottom
        self.menu_frame = tk.Frame(self, background="dark gray")
        self.menu_frame.pack(
            fill=tk.BOTH,
            side=tk.TOP,
            expand=True,
        )

        # Create a container for buttons and center it
        button_container = tk.Frame(self.menu_frame, background="dark gray")
        button_container.pack(anchor="center")

        # Container frame to hold different pages
        self.container = tk.Frame(self)
        self.container.pack(expand=True, fill=tk.BOTH, side=tk.TOP)

        selector_button = tk.Button(
            button_container,
            text="Go to Layout Selector",
            command=lambda: self.show_frame(LayoutSelectorApp),
            width=30,
        )
        selector_button.grid(row=0, column=0, padx=20, pady=10)

        creator_button = tk.Button(
            button_container,
            text="Go to Layout Creator",
            command=lambda: self.show_frame(LayoutCreatorApp),
            width=30,
        )
        creator_button.grid(row=0, column=1, padx=20, pady=10)

        vehicle_button = tk.Button(
            button_container,
            text="Go to Vehicle Editor",
            command=lambda: self.show_frame(VehicleEditorApp),
            width=30,
        )
        vehicle_button.grid(row=0, column=2, padx=20, pady=10)

        help_button = tk.Button(
            button_container,
            text="Help",
            command=lambda: self.show_frame(HelpFrame),
            width=30,
        )
        help_button.grid(row=0, column=3, padx=40, pady=10)

        # Create frames for each app
        self.frames = {}
        for F in (LayoutSelectorApp, LayoutCreatorApp, VehicleEditorApp, HelpFrame):
            frame = None
            if F == LayoutSelectorApp:
                frame = F(self.container, folder_path=self.folder_path)
            else:
                frame = F(self.container)
            self.frames[F] = frame
            frame.grid(row=1, column=0, sticky="nsew")

        self.show_frame(LayoutCreatorApp)  # Show the layout selector by default
        # Add this line to bind the close protocol
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def configure_global_font(self, font_family="Consolas", font_size=10):
        """Apply global font settings to all widgets."""
        default_font = (font_family, font_size)
        # Set default font for all standard Tkinter widgets
        self.option_add("*Font", default_font)

        # Configure ttk widgets separately
        style = ttk.Style()
        style.configure("TButton", font=default_font)  # ttk.Button
        style.configure("TLabel", font=default_font)  # ttk.Label

    def show_frame(self, frame_class):
        """Show the selected frame."""
        frame = self.frames[frame_class]
        if isinstance(frame, LayoutSelectorApp):
            frame.refresh_files()
        frame.tkraise()

    def on_close(self):
        """Handle the close event to stop the program."""
        self.quit()  # Stop the Tkinter main loop
        self.destroy()  # Destroy the window and release resources
        sys.exit()  # Exit the Python interpreter


if __name__ == "__main__":
    app = CARLOSCompanion()
    app.mainloop()
