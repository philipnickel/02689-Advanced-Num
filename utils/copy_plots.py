#!/usr/bin/env python3
"""
Script to copy the entire Plots/ directory from assignment_1/ to
../Advanced-Numerical-Methods---Assignment-1/
"""

import os
import shutil
import sys
from pathlib import Path


def main():
    """Copy Plots directory to target location, overwriting if it exists."""

    # Get the script's directory and project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Source and destination paths
    source_plots = project_root / "assignment_1" / "Plots"
    dest_dir = project_root.parent / "Advanced-Numerical-Methods---Assignment-1"
    dest_plots = dest_dir / "Plots"

    # Check if source exists
    if not source_plots.exists():
        print(f"Error: Source directory {source_plots} does not exist")
        sys.exit(1)

    # Check if destination parent directory exists
    if not dest_dir.exists():
        print(f"Error: Destination directory {dest_dir} does not exist")
        sys.exit(1)

    try:
        # Remove destination if it exists
        if dest_plots.exists():
            print(f"Removing existing {dest_plots}")
            shutil.rmtree(dest_plots)

        # Copy the entire directory tree
        print(f"Copying {source_plots} to {dest_plots}")
        shutil.copytree(source_plots, dest_plots)

        print("Successfully copied Plots directory")

    except Exception as e:
        print(f"Error copying plots: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()