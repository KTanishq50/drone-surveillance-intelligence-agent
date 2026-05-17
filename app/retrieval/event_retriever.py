from app.retrieval.retriever_base import filter_by_threshold
from app.utils.logger import get_logger

logger = get_logger("event_retriever")

def retrieve(observation: dict, memory_hub) -> list:
    results = memory_hub.vector.query_events(observation["embedding"], top_k=3)
    filtered = filter_by_threshold(results)
    logger.debug(f"Event retrieval: {len(filtered)} similar events found")
    return filtered