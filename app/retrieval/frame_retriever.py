from app.retrieval.retriever_base import filter_by_threshold
from app.utils.logger import get_logger

logger = get_logger("frame_retriever")

def retrieve(observation: dict, memory_hub) -> list:
    results = memory_hub.vector.query_frames(observation["embedding"], top_k=5)
    filtered = filter_by_threshold(results)
    logger.debug(f"Frame retrieval: {len(filtered)} similar frames found")
    return filtered