from app.engine.event_synthesizer import synthesize
from app.utils.logger import get_logger

logger = get_logger("synthesis_worker")

async def run(observation: dict, memory_hub) -> dict:
    logger.debug(f"Synthesis worker starting for frame {observation['frame_id']}")
    event = synthesize(observation, memory_hub)
    return {
        "frame_id": observation["frame_id"],
        "worker":   "synthesis",
        "event":    event,
    }