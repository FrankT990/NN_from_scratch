import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import json
import os
import subprocess 
from threading import Thread 
import time
import re


window = tk.Tk()
window.config(bg="black")
window.title("Neural Network From Scratch")
window.geometry("1100x800")

light_green = "#00ff03"

#  generated functions for C++ calls
def show_loading_popup(title):
    """
    Displays a simple, non-blocking 'Loading...' popup.
    This function now accesses the global 'window' and 'light_green' variables.
    """
    # Access global variables
    global window, light_green
    
    loading_popup = tk.Toplevel(window)
    loading_popup.title(title)
    loading_popup.config(bg="black")
    
    # Center the popup on the main window
    window_x = window.winfo_x()
    window_y = window.winfo_y()
    window_width = window.winfo_width()
    window_height = window.winfo_height()
    
    popup_width = 200
    popup_height = 100
    
    x = window_x + (window_width // 2) - (popup_width // 2)
    y = window_y + (window_height // 2) - (popup_height // 2)
    
    loading_popup.geometry(f'{popup_width}x{popup_height}+{x}+{y}')
    
    loading_label = tk.Label(
        loading_popup,
        text="training in progress",
        font=("Arial", 12),
        bg="black",
        fg=light_green # Use global color
    )
    loading_label.pack(expand=True, padx=20, pady=20)
    
    loading_popup.transient(window)
    loading_popup.grab_set()
    loading_popup.update()
    return loading_popup

def run_cpp_command_in_thread(command, on_complete_callback):
    """
    Runs a subprocess command in a separate thread to avoid freezing the GUI.
    This function now accesses the global 'window' variable.
    """
    # Access global variable
    global window
    
    def thread_target():
        try:
            # Get the directory where this Python script (App.py) is running
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            executable_name = 'main.exe' if os.name == 'nt' else 'main'
            executable_path = os.path.join(script_dir, executable_name)
            
            full_command = [executable_path] + command
            
            print(f"--- Debug: Running command ---")
            print(f"Full command: {' '.join(full_command)}")

            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                check=False,
                encoding='utf-8' # Explicitly set encoding
            )
            
            print(f"--- Debug: Process finished ---")
            print(f"Return Code: {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            print(f"-------------------------------")

        except Exception as e:
            # Create a mock result object for exceptions (e.g., file not found)
            class MockResult:
                def __init__(self):
                    self.returncode = -1
                    self.stdout = ""
                    self.stderr = (
                        f"Python Error: {e}\n\n"
                        f"Failed to execute command.\n"
                        f"Please ensure '{executable_name}' is compiled and located in the same directory as your App.py script:\n"
                        f"{script_dir}"
                    )
            
            result = MockResult()
            
        # Schedule the callback to run on the main GUI thread
        window.after(0, on_complete_callback, result)

    # Start the thread
    Thread(target=thread_target, daemon=True).start()

def test_single_image():
    """
    Asks the user for a single PNG file, runs the C++ model in 'predict' mode,
    and shows the result in a popup. Takes 0 arguments.
    """
    print("--- test_single_image() CALLED ---")
    
    model_file = "model.json"
    config_file = "config.json"
    
    if not os.path.exists(model_file) or not os.path.exists(config_file):
        print(f"--- DEBUG: File check failed. model.json exists: {os.path.exists(model_file)}, config.json exists: {os.path.exists(config_file)}")
        messagebox.showerror(
            "Error",
            f"Files not found!\n\n"
            f"Please ensure both 'model.json' and 'config.json' exist in the same directory.\n"
            f"You may need to train the network first."
        )
        return
    
    print("--- DEBUG: File check passed. Opening file dialog... ---")

    file_path = filedialog.askopenfilename(
        title="Select PNG Image for Prediction",
        filetypes=[("PNG files", "*.png")]
    )
    
    if not file_path:
        print("--- DEBUG: File dialog cancelled. ---")
        return  # User cancelled
        
    print(f"--- DEBUG: File selected: {file_path} ---")

    loading_popup = show_loading_popup("Predicting") # <-- This now works

    def on_prediction_complete(result):
        """Callback function to handle the result from the C++ executable."""
        loading_popup.destroy()
        
        if result.returncode != 0:
            messagebox.showerror(
                "Prediction Error",
                f"The model failed to make a prediction.\n\nError:\n{result.stderr}"
            )
            return
        
        # Parse the output
        output = result.stdout
        try:
            # Use regex to find the prediction and confidence
            prediction_match = re.search(r"Prediction: (.*)", output)
            confidence_match = re.search(r"Confidence: (.*)%", output)
            
            if not prediction_match or not confidence_match:
                raise ValueError("Could not parse prediction output.")
                
            prediction = prediction_match.group(1).strip()
            confidence = confidence_match.group(1).strip()
            
            messagebox.showinfo(
                "Prediction Result",
                f"Prediction: {prediction}\nConfidence: {confidence}%"
            )
            
        except Exception as e:
            messagebox.showerror(
                "Parsing Error",
                f"Could not parse the model's output.\n\nError: {e}\n\nFull Output:\n{output}"
            )

    # Run the C++ executable in '--predict' mode
    run_cpp_command_in_thread(
        command=['--predict', file_path],
        on_complete_callback=on_prediction_complete
    )


def test_dataset():
    """
    Asks the user for a test dataset folder, runs the C++ model in 'test' mode,
    and shows the accuracy in a popup. Takes 0 arguments.
    """
    print("--- test_dataset() CALLED ---")
    
    model_file = "model.json"
    config_file = "config.json"

    if not os.path.exists(model_file) or not os.path.exists(config_file):
        print(f"--- DEBUG: File check failed. model.json exists: {os.path.exists(model_file)}, config.json exists: {os.path.exists(config_file)}")
        messagebox.showerror(
            "Error",
            f"Files not found!\n\n"
            f"Please ensure both 'model.json' and 'config.json' exist in the same directory.\n"
            f"You may need to train the network first."
        )
        return
    
    print("--- DEBUG: File check passed. Opening folder dialog... ---")

    folder_path = filedialog.askdirectory(
        title="Select Test Dataset Folder"
    )
    
    if not folder_path:
        print("--- DEBUG: Folder dialog cancelled. ---")
        return  # User cancelled
        
    print(f"--- DEBUG: Folder selected: {folder_path} ---")

    loading_popup = show_loading_popup("Testing") # <-- This now works

    def on_test_complete(result):
        """Callback function to handle the result from the C++ executable."""
        loading_popup.destroy()
        
        if result.returncode != 0:
            messagebox.showerror(
                "Testing Error",
                f"The model failed during testing.\n\nError:\n{result.stderr}"
            )
            return
            
        # Parse the output
        output = result.stdout
        try:
            # Use regex to find the accuracy
            accuracy_match = re.search(r"Accuracy: (.*)%", output)
            
            if not accuracy_match: # <-- Fixed a typo here
                raise ValueError("Could not parse testing output.")
                
            accuracy = accuracy_match.group(1).strip()
            
            messagebox.showinfo(
                "Test Result",
                f"Test Accuracy: {accuracy}%"
            )
            
        except Exception as e:
            messagebox.showerror(
                "Parsing Error",
                f"Could not parse the model's output.\n\nError: {e}\n\nFull Output:\n{output}"
            )

    # Run the C++ executable in '--test' mode
    run_cpp_command_in_thread(
        command=['--test', folder_path],
        on_complete_callback=on_test_complete
    )

def run_cpp_training(config_data):
    """
    Shows a 'Training...' popup and runs the C++ 'main --train' command
    in a separate thread.
    """
    print("Starting C++ training...")
    loading_popup = show_loading_popup("Training") # <-- This now works
    
    def on_training_complete(result):
        """Callback function to run on the main thread when training is done."""
        loading_popup.destroy()
        if result.returncode == 0:
            messagebox.showinfo(
                "Training Complete",
                "Successfully trained network and saved 'model.json'."
            )
            print("Training successful.")
        else:
            messagebox.showerror(
                "Training Failed",
                f"The C++ training process failed.\n\nError:\n{result.stderr}"
            )
            print(f"Training failed. STDOUT: {result.stdout}, STDERR: {result.stderr}")
            
    # Run 'main --train' in a separate thread
    run_cpp_command_in_thread(
        command=['--train'],
        on_complete_callback=on_training_complete
    )

def generate_config():
    """Validates inputs and generates the config.json file."""
    global layers, dataset_path, output_activation
    
    print("Generating config...")
    
    # --- 1. Validate Inputs ---
    if not dataset_path:
        messagebox.showerror("Error", "Please select a dataset path.")
        return
        
    if not layers:
        messagebox.showerror("Error", "Please add at least one layer.")
        return
        
    if layers[-1].get('type') != 'output':
        messagebox.showerror("Error", "Network is missing an output layer. This is a bug.")
        return

    # --- 2. Get Layer Data from UI ---
    # Final check on output layer
    layers[-1]['activation'] = output_activation.get()
    
    # --- 3. Create JSON Structure ---
    config_data = {
        "dataset": dataset_path,
        "layers": layers,
        "output_activation": output_activation.get()
    }
    
    # --- 4. Write to file ---
    try:
        with open("config.json", "w") as f:
            json.dump(config_data, f, indent=4)
        
        print(f"Successfully wrote config.json")
        
        # --- FIX: Wait for 0.2 seconds ---
        time.sleep(0.2)
        
        # --- 5. Run the C++ training ---
        run_cpp_training(config_data)
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to write config.json: {e}")
        # Print the *actual* error to the console for debugging
        print(f"Error during config generation/training startup: {e}")
        import traceback
        traceback.print_exc()

dataset_path = None
layers = []
output_activation = tk.StringVar(value="softmax")

rect_width = 900   # Increase width
rect_height = 500  # Increase height

canvas_width = 900
canvas_height = 500

canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="black", highlightthickness=0)
canvas.grid(row=4, column=0, columnspan=10, sticky="nsew", padx=0, pady=(0, 20))

rect_id = None  

def draw_centered_rectangle(event=None):
    """(Re)draw the centered rectangle and then redraw the network."""
    global rect_id
    canvas.delete("all")
    w = canvas.winfo_width()
    h = canvas.winfo_height()
    rw = min(rect_width, w - 20)
    rh = min(rect_height, h - 20)
    x0 = (w - rw) // 2
    y0 = (h - rh) // 2 + 40  # Slightly down
    x1 = x0 + rw
    y1 = y0 + rh
    rect_id = canvas.create_rectangle(x0, y0, x1, y1, outline=light_green, width=3, tags="rect")
    
    update_layers_display()


canvas.bind("<Configure>", draw_centered_rectangle)

# Functions

def choose_dataset():
    global dataset_path
    folder_path = filedialog.askdirectory()
    if folder_path:
        dataset_path = folder_path.replace("\\", "/")
        display_text = dataset_path
        if len(display_text) > 40:
            display_text = display_text[0:25] + "..."
        label_file_selected.config(text=f"Folder chosen:\n{display_text}")

        # Detect subfolders (classes)
        class_folders = [
            name for name in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, name))
        ]
        class_folders.sort()  # keep order consistent

        if class_folders:
            # Remove any existing output layer
            global layers
            layers = [l for l in layers if l.get("type") != "output"]

            # Create new output layer
            output_layer = {
                "type": "output",
                "nodes": len(class_folders),
                "classes": class_folders,
                "activation": output_activation.get()
            }
            layers.append(output_layer)

        update_layers_display()

def add_hidden_layer():
    popup = tk.Toplevel(window)
    popup.title("Add Hidden Layer")

    tk.Label(popup, text="Number of Nodes:", fg=light_green, bg="black", font=("Arial", 11)).pack(pady=5)
    nodes_var = tk.StringVar(value="64")
    ttk.Combobox(popup, textvariable=nodes_var, values=["8", "16", "32", "64", "128", "256"], state="readonly").pack(
        pady=5
    )

    tk.Label(popup, text="Activation Function:", fg=light_green, bg="black", font=("Arial", 11)).pack(pady=5)
    activation_var = tk.StringVar(value="relu")
    ttk.Combobox(popup, textvariable=activation_var, values=["relu", "sigmoid", "tanh"], state="readonly").pack(
        pady=5
    )

    def save_layer():
        layer = {"nodes": int(nodes_var.get()), "activation": activation_var.get()}
        layers.append(layer)
        update_layers_display()
        popup.destroy()

    tk.Button(
        popup,
        text="Save Layer",
        command=save_layer,
        font=("Arial", 11, "bold"),
        borderwidth=2,
        relief="solid",
    ).pack(pady=10)


def remove_last_layer():
    if layers:
        layers.pop()
        update_layers_display()
    else:
        messagebox.showwarning("Warning", "No layers to remove!")


def update_layers_display():
    """Draw the layer text list and the network (vertical node rows + fully connected lines)."""
    # Clear layer text frame
    for widget in layers_frame.winfo_children():
        widget.destroy()
    
    canvas.delete("layer")

    # If no layers or rect not yet created, nothing more to draw
    if not layers or rect_id is None:
        return

    # Rectangle bounds
    rect_x0, rect_y0, rect_x1, rect_y1 = canvas.coords(rect_id)
    rect_width_current = rect_x1 - rect_x0
    rect_height_current = rect_y1 - rect_y0

    # Spacing between layers
    num_layers = len(layers)
    x_spacing = rect_width_current / (num_layers + 1)

    all_positions = []  # store (x, y) positions for each layer's nodes

    # Draw nodes
    for i, layer in enumerate(layers):
        num_nodes = layer["nodes"]
        cx = rect_x0 + (i + 1) * x_spacing  # x for this layer

        layer_positions = []

        if num_nodes in (64, 128, 256):
            total_drawn = 16 + 1
            y_spacing = rect_height_current / (total_drawn + 2)  # spacing with margin

            # --- Draw top 16 nodes ---
            for j in range(16):
                cy = rect_y0 + (j + 1) * y_spacing
                r = 6
                canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                        outline=light_green, fill="", width=2, tags="layer")
                layer_positions.append((cx, cy))

            # --- Ellipsis in the middle ---
            ellipsis_y = rect_y0 + (17) * y_spacing
            canvas.create_text(cx, ellipsis_y, text="⋮",
                        fill=light_green, font=("Arial", 24), tags="layer")

            # --- Bottom node ---
            cy = rect_y1 - y_spacing
            r = 6
            canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                    outline=light_green, fill="", width=2, tags="layer")
            layer_positions.append((cx, cy))

        else:
            # Normal representation (all nodes drawn)
            y_spacing = rect_height_current / (num_nodes + 1)
            for j in range(num_nodes):
                cy = rect_y0 + (j + 1) * y_spacing
                r = 6
                canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                        outline=light_green, fill="", width=2, tags="layer")
                layer_positions.append((cx, cy))

                if layer.get("type") == "output":
                    class_name = layer["classes"][j]
                    canvas.create_text(
                        cx + 20, cy,
                        text=class_name,
                        fill=light_green,
                        font=("Arial", 9),
                        anchor="w",
                        tags="layer"
                    )
        all_positions.append(layer_positions)

    for i in range(len(all_positions) - 1):
        for (x1, y1) in all_positions[i]:
            for (x2, y2) in all_positions[i + 1]:
                canvas.create_line(x1, y1, x2, y2, fill=light_green, width=1, tags="layer")


# Title
label_title = tk.Label(text="Neural Network From Scratch", fg=light_green, bg="black", font=("Arial", 24, "bold"))
label_title.grid(row=0, column=0, columnspan=10, sticky="new", pady=(20, 0))

label_author = tk.Label(text="Frank Tittiger", fg=light_green, bg="black", font=("Arial", 12))
label_author.grid(row=1, column=0, columnspan=10, sticky="n", pady=(0, 20))

# Upload training data
btn_upload_dataset = tk.Button(
    master=window,
    text="Upload Dataset",
    command=choose_dataset,
    height=2,
    width=18,
    font=("Arial", 12),
    borderwidth=2,
    relief="solid",
)
btn_upload_dataset.grid(row=2, column=3, sticky="ew", padx=10, pady=(0, 0))

# File selected label
label_file_selected = tk.Label(master=window, text="Folder chosen:\nNo folder chosen yet", bg="black", fg=light_green, font=("Arial", 11))
label_file_selected.grid(row=3, column=3, sticky="ew", padx=10, pady=(5, 20))

# adding hidden layers
btn_add_layer = tk.Button(
    master=window,
    text="Add Hidden Layer",
    command=add_hidden_layer,
    height=2,
    width=18,
    font=("Arial", 12),
    borderwidth=2,
    relief="solid",
)
btn_add_layer.grid(row=2, column=5, sticky="ew", padx=10, pady=(0, 0))

btn_remove_layer = tk.Button(master=window, text="Remove Last Layer", command=remove_last_layer, height=2, width=18, font=("Arial", 12), borderwidth=2, relief="solid")
btn_remove_layer.grid(row=3, column=5, sticky="ew", padx=10, pady=(0, 0))

layers_frame = tk.Frame(window, bg="black")
layers_frame.grid(row=3, column=6, sticky="n", padx=10, pady=(5, 20))

# output 
label_output = tk.Label(text="Output Activation:", fg=light_green, bg="black", font=("Arial", 12))
label_output.grid(row=2, column=7, sticky="s", padx=10)
output_dropdown = ttk.Combobox(window, textvariable=output_activation, values=["softmax"], state="readonly")
output_dropdown.grid(row=3, column=7, sticky="n", padx=10)

# make config
btn_train = tk.Button(
    master=window,
    text="Train Neural Network",
    command=generate_config,
    height=2,
    width=20,
    font=("Arial", 12, "bold"),
    borderwidth=2,
    relief="solid",
)
btn_train.grid(row=2, column=8, sticky="ew", padx=10, pady=(0, 5))

# predict single image
btn_predict = tk.Button(
    master=window,
    text="Predict Single Image",
    command=test_single_image, 
    height=2,
    width=20,
    font=("Arial", 12, "bold"),
    borderwidth=2,
    relief="solid",
)
btn_predict.grid(row=3, column=8, sticky="ew", padx=10, pady=(0, 5))

# test on dataset
btn_test_dataset = tk.Button(
    master=window,
    text="Test on Dataset",
    command= test_dataset, # <-- UPDATED
    height=6,
    width=20,
    font=("Arial", 12, "bold"),
    borderwidth=2,
    relief="solid",
)
# btn_test_dataset.grid(row=4, column=8, sticky="ew", padx=10, pady=(0, 5)) 

for i in range(10):
    window.columnconfigure(i, weight=1)
window.rowconfigure(4, weight=1)

# force an initial draw of the rectangle (so rect_id exists before first add)
window.update_idletasks()
draw_centered_rectangle()



window.mainloop()