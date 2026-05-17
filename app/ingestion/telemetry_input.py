import json
from app.utils.logger import get_logger

logger = get_logger("telemetry_input")

def load_telemetry(path: str) -> dict:
    with open(path, "r") as f:
        entries = json.load(f)
    indexed = {e["frame_id"]: e for e in entries}
    logger.info(f"Loaded telemetry for {len(indexed)} frames from {path}")
    return indexed