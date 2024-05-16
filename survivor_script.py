import os
import subprocess

# Get the current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))
survivors_path = os.path.join(current_directory, 'survivor.pt')
source_path = os.path.join(current_directory, 'eqvid.mp4')

# Command to run the survivor detection
survivor_command = [
    'python',
    'survivor_detect.py',
    '--weights', survivors_path,
    '--source', source_path, #1 koi si bhi image #2 processed image of flood
    '--img-size', '640',
    '--delt', '2', #delt=1 for whole image & delt=2 for only flood boxes
    '--type', 'Flood'
]

# Run the survivor detection command
subprocess.run(survivor_command, shell=True)