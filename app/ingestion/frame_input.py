import json
from app.utils.logger import get_logger

logger = get_logger("frame_input")

def load_frames(path: str) -> list:
    with open(path, "r") as f:
        frames = json.load(f)
    logger.info(f"Loaded {len(frames)} frames from {path}")
    return frames