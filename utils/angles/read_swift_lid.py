import subprocess
from pathlib import Path
import utils.config as config

def read_swift_lid():
    # Define the Swift file paths relative to the repo root.
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "utils" / "angles"
    lid_angle_path = path / "lid_angle.swift"
    lid_hardware_path = path / "HardwareCompat.swift"


    command = f"swiftc {lid_angle_path} {lid_hardware_path} -o /tmp/lid_angle && /tmp/lid_angle"
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        config.lid_angle = float(result.stdout.strip())
    
    except subprocess.CalledProcessError as e:
        print(f"Error reading LID file: {e}")
        return None
