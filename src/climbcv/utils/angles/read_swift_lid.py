import subprocess
import re
from pathlib import Path
import time

def read_swift_lid(lid_angle_value, lid_timestamp, stop_event=None, poll_interval: float = 0.5) -> None:

    OVERRIDE_COMPILED = True # For testing
    
    repo_root: Path = Path(__file__).resolve().parents[4]


    build_dir: Path = repo_root / "build"
    build_path: Path = build_dir / "LidAngle_Compiled"
    path: Path = repo_root / "src" / "climbcv" / "utils" / "angles"
    command: str = None

    if not build_path.exists() and not OVERRIDE_COMPILED:

        if not build_dir.exists():
            build_dir.mkdir(parents=True)

        print("Compiling LidAngle from switft to binary")
        command = f"swiftc {path / 'lid_angle.swift'} {path / 'hardware_compat.swift'} -o {build_path}"
        subprocess.run(command, shell=True, check=True)


    if not OVERRIDE_COMPILED:
        command = f"{build_path}"
    else:
        lid_angle_path: Path = path / "lid_angle.swift"
        lid_hardware_path: Path = path / "hardware_compat.swift"
        command = f"swiftc {lid_angle_path} {lid_hardware_path} -o /tmp/lid_angle && /tmp/lid_angle"

        
    while stop_event is None or not stop_event.is_set():
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

        if stop_event is not None and stop_event.is_set():
            break

        time.sleep(poll_interval)

    return None