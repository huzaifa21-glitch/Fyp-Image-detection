import os
import subprocess

# Get the current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Detect folder path
detect_folder = os.path.join(current_directory, 'runs', 'detect')

# Relative paths from the current working directory
earthquake_path = os.path.join(current_directory, 'best.pt')
survivors_path = os.path.join(current_directory, 'survivor.pt')
source_path = os.path.join(current_directory, 'photo9.jpg')
source_path2 = os.path.join(current_directory, 'temp', 'tempEarth.jpg')

# Command to run the earthquake detection
earthquake_command = [
    'python',
    'earthquake_track.py',
    '--weights', earthquake_path,
    '--source', source_path,
    '--img-size', '640',
    '--delt', '1',
    '--city', 'Gaza'
]

# Run the earthquake detection command
subprocess.run(earthquake_command, shell=True)


