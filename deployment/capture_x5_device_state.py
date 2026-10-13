"""Read-only deployment telemetry; use GPU-scope power readings on RTX laptops."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def snapshot():
    root = ET.fromstring(subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True))
    gpu = root.find("gpu")
    result = {"utc": datetime.now(timezone.utc).isoformat(), "gpu": gpu.findtext("product_name"),
              "driver": root.findtext("driver_version")}
    for name in ("gpu_power_readings", "temperature", "clocks", "clocks_event_reasons"):
        node = gpu.find(name)
        result[name] = {} if node is None else {x.tag: x.text for x in node}
    profile = Path("/sys/firmware/acpi/platform_profile")
    result["platform_profile"] = profile.read_text().strip() if profile.exists() else None
    try:
        result["power_profile"] = subprocess.check_output(["powerprofilesctl", "get"], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        result["power_profile"] = None
    return result


if __name__ == "__main__":
    print(json.dumps(snapshot(), indent=2))
