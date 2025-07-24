import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext


class HelpFrame(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, background="light blue", *args, **kwargs)

        self.file_path = "./gui_help_files/"
        main_help_filename = "gui_help.txt"

        main_label = tk.Label(
            self,
            text=self.read_help_file(main_help_filename)["title"],
            justify="center",
            background="light blue",
        )
        main_label.pack(fill=tk.X, side=tk.TOP, pady=10)

        creator_options = self.read_help_file("layout_creator_help.txt")
        selector_options = self.read_help_file("layout_selector_help.txt")
        vehicle_options = self.read_help_file("vehicle_editor_help.txt")

        self.current_frame = "selector"

        self.frame_dicts = {
            "creator": {
                "title": creator_options.pop("title"),
                "options": creator_options,
            },
            "selector": {
                "title": selector_options.pop("title"),
                "options": selector_options,
            },
            "vehicle": {
                "title": vehicle_options.pop("title"),
                "options": vehicle_options,
            },
        }

        # Frame to hold the creator_frame or selector_frame
        main_frame = tk.Frame(self, background="light blue")
        main_frame.pack(expand=True, side=tk.TOP, fill=tk.BOTH)

        # Button frame to hold the creator and selector buttons
        button_frame = tk.Frame(main_frame, background="light blue")
        button_frame.pack(anchor="center", side=tk.TOP, pady=10, fill=tk.Y)
        # Button to show creator_frame - exists in the main_frame
        creator_button = tk.Button(
            button_frame,
            text="Layout Creator Help",
            command=lambda: self.show_help_frame("creator"),
        )
        creator_button.grid(row=0, column=0, padx=20)
        # Button to show selector_frame - exists in the main frame
        selector_button = tk.Button(
            button_frame,
            text="Layout Selector Help",
            command=lambda: self.show_help_frame("selector"),
        )
        selector_button.grid(row=0, column=1, padx=20)
        # Button to show vehicle_frame - exists in the main frame
        selector_button = tk.Button(
            button_frame,
            text="Vehicle Editor Help",
            command=lambda: self.show_help_frame("vehicle"),
        )
        selector_button.grid(row=0, column=2, padx=20)

        # Frame to hold the help options
        options_frame = tk.Frame(main_frame, background="light blue")
        options_frame.pack(
            side=tk.TOP, anchor="center", pady=10, fill=tk.BOTH, expand=True
        )

        # Main label for creator_frame
        self.frame_label = tk.Label(
            options_frame,
            text="Nothing Selected",
            justify="center",
            height=3,
            background="light blue",
        )
        self.frame_label.pack(anchor="center", side=tk.TOP, pady=5, ipady=10, fill=tk.X)

        list_frame = tk.Frame(options_frame, background="light blue")
        list_frame.pack(side=tk.TOP, anchor="center", fill=tk.BOTH, expand=True)

        # List selection widget
        self.list_widget = tk.Listbox(
            list_frame, height=28, width=30, border=2, relief="ridge"
        )
        self.list_widget.pack(
            side=tk.LEFT, expand=True, anchor="center", fill=tk.BOTH, padx=10, pady=10
        )
        # Bind the selection event
        self.list_widget.bind("<<ListboxSelect>>", self.display_option)

        # Label to display the selected option
        self.option_label = scrolledtext.ScrolledText(
            list_frame,
            height=30,
            width=100,
            background="white",
            wrap="word",
            border=2,
            relief="ridge",
            spacing1=3,
        )
        self.option_label.pack(
            side=tk.RIGHT, expand=True, pady=10, fill=tk.BOTH, padx=10
        )

        # # Scrollbar for the option label
        # scrollbar = tk.Scrollbar(self.option_label, orient="vertical", command=self.option_label.yview)
        # scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def show_help_frame(self, name):
        self.selected_frame = name
        self.list_widget.delete(0, tk.END)
        for key in self.frame_dicts[name]["options"]:
            self.list_widget.insert(tk.END, key)
        self.frame_label.config(text=self.frame_dicts[name]["title"])

    def display_option(self, event):
        """Load the selected file and display its layout."""
        selected_index = (
            self.list_widget.curselection()
        )  # Get the index of the selected file
        option_name = self.list_widget.get(selected_index)
        self.option_label.delete(1.0, tk.END)
        if not selected_index:
            self.option_label.insert(tk.END, "No option selected")
        else:
            selected_option = self.frame_dicts[self.selected_frame]["options"][
                option_name
            ]
            self.option_label.insert(tk.END, selected_option)

    def read_help_file(self, filename):
        filename = self.file_path + filename
        options = {}
        with open(filename, "r") as f:
            current_option = "title"
            options[current_option] = ""
            for line in f:
                if line[0] == "$":
                    current_option = line.strip()[1:]
                    options[current_option] = ""
                    continue
                else:
                    options[current_option] += line
        return options
