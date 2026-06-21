import subprocess
import re
from pathlib import Path
import time

def read_swift_lid(lid_angle_value, lid_timestamp, stop_event=None, poll_interval: float = 0.5) -> None:

    # When True, force a fresh compile on startup even if a cached binary
    # exists (handy while editing the Swift source). Either way the compile
    # happens once, before the poll loop, not on every poll.
    OVERRIDE_COMPILED = True

    repo_root: Path = Path(__file__).resolve().parents[4]

    build_dir: Path = repo_root / "build"
    build_path: Path = build_dir / "LidAngle_Compiled"
    path: Path = repo_root / "src" / "climbcv" / "utils" / "angles"
    command: str = None

    if not build_path.exists() and not OVERRIDE_COMPILED:

        if not build_dir.exists():
            build_dir.mkdir(parents=True)

        print("Compiling LidAngle from swift to binary")
        command = f"swiftc {path / 'lid_angle.swift'} {path / 'hardware_compat.swift'} -o {build_path}"
        subprocess.run(command, shell=True, check=True)

    # Compile the Swift sources to a binary once, up front.
    if OVERRIDE_COMPILED or not build_path.exists():
        build_dir.mkdir(parents=True, exist_ok=True)

        lid_angle_path: Path = path / "lid_angle.swift"
        lid_hardware_path: Path = path / "hardware_compat.swift"

        print("Compiling LidAngle from swift to binary")
        compile_command = f"swiftc {lid_angle_path} {lid_hardware_path} -o {build_path}"
        try:
            subprocess.run(compile_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error compiling LidAngle: {e}")
            return

    command: str = f"{build_path}"

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
