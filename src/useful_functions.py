import os
import warnings
import re

def read_dirs_paths(file_path, global_scope):
    """
    Reads a text file where each line contains a variable assignment
    and creates those variables in the global scope.
    Also prints all created variables.

    Args:
        file_path (str): Path to the text file.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        warnings.warn(f"File '{file_path}' not found. No variables were created.", UserWarning)
        return

    created_variables = {}  # Dictionary to store created variables and their values

    # Open the file and read it line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and split the line at '='
            line = line.strip()
            if '=' in line:
                var_name, var_value = line.split('=', 1)
                var_name = var_name.strip()  # Remove any extra spaces around the variable name
                var_value = var_value.strip().strip("'")  # Remove spaces around the value
                
                # Retain quotes for string values (consider values between single quotes)
                #if var_value.startswith("'") and var_value.endswith("'"):
                #    var_value = f"'{var_value[1:-1]}'"  # Keep the quotes in the string value
                
                # Use the provided global_scope to create the variable
                global_scope[var_name] = var_value
                created_variables[var_name] = var_value  # Store the variable in the dictionary

    # Print the created variables
    if created_variables:
        print("Created variables:")
        for var_name, var_value in created_variables.items():
            print(f"{var_name} = {var_value}")
    else:
        print("No variables were created.")

def check_directories(out_dir,out_trajectories2,out_trajectories3,out_trajectories4):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir )
        os.makedirs(out_trajectories2)
        os.makedirs(out_trajectories3)
        os.makedirs(out_trajectories4)
        print(out_dir, "created successfully!")
        print(out_trajectories2, "created successfully!")
        print(out_trajectories3, "created successfully!")
        print(out_trajectories4, "created successfully!")
    else:
        print(out_dir, "already exists!")

    if not os.path.exists(out_trajectories2):
        os.makedirs(out_trajectories2)
        os.makedirs(out_trajectories3)
        os.makedirs(out_trajectories4)
        print(out_dir, "created successfully!")
        print(out_trajectories2, "created successfully!")
        print(out_trajectories3, "created successfully!")
        print(out_trajectories4, "created successfully!")
    else:
        print(out_trajectories2, "already exists!")

    if not os.path.exists(out_trajectories3):
        os.makedirs(out_trajectories3)
        print(out_trajectories3, "created successfully!")
    else:
        print(out_trajectories3, "already exists!")

    if not os.path.exists(out_trajectories4):
        os.makedirs(out_trajectories4)
        print(out_trajectories4, "created successfully!")
    else:
        print(out_trajectories4, "already exists!")

    print(" ")
    print(" ")


def get_latest_state_file(folder_path):
    # Regex to match files like x0_123.xml
    pattern = re.compile(r"x0_(\d+)\.xml$")

    latest_file = None
    latest_num = -1

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))
            if num > latest_num:
                latest_num = num
                latest_file = filename

    if latest_file:
        return os.path.join(folder_path, latest_file)
    else:
        return None
