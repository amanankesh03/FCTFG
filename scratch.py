import sys
import os

# Get the current directory (assuming Trainer.py is in the project root)
project_base_path = os.path.abspath(os.path.dirname(__file__))

# Add the project base path to the Python path
sys.path.append(project_base_path)