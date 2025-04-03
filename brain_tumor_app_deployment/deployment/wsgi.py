import sys
import os

# Add your project directory to the sys.path
path = '/home/yourusername/brain_tumor_app'
if path not in sys.path:
    sys.path.append(path)

# Import your Flask app
from app import app as application

# Make sure directories exist
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads'), exist_ok=True)
