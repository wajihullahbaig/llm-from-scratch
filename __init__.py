import sys
import os
from pathlib import Path

# Get absolute path of this file's directory (project root)
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT 

# Add paths to Python path if not already present
paths = [str(PROJECT_ROOT), str(SRC_PATH)]
for path in paths:
    if path not in sys.path:
        sys.path.append(path)