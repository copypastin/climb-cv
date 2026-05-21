import subprocess
import re
from pathlib import Path
import time

def read_swift_lid(lid_angle_value, lid_timestamp):
    # Define the Swift file paths relative to the repo root.
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "utils" / "angles"
    lid_angle_path = path / "lid_angle.swift"
    lid_hardware_path = path / "HardwareCompat.swift"


    command = f"swiftc {lid_angle_path} {lid_hardware_path} -o /tmp/lid_angle && /tmp/lid_angle"
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        match = re.search(r"[-+]?\d*\.?\d+", result.stdout)
        if match:
            lid_angle_value.value = float(match.group(0))
            lid_timestamp.value = time.time()
        else:
            print(f"Error reading mac angle output: {result.stdout.strip()}")
    
    except subprocess.CalledProcessError as e:
        print(f"Error reading mac angle file: {e}")
        lid_angle_value.value = None
        lid_timestamp.value = None
        return None
