from app.retrieval.retriever_base import filter_by_threshold
from app.utils.logger import get_logger

logger = get_logger("chain_retriever")

def retrieve(observation: dict, memory_hub) -> list:
    results = memory_hub.vector.query_chains(observation["embedding"], top_k=3)
    filtered = filter_by_threshold(results, floor=0.45)
    logger.debug(f"Chain retrieval: {len(filtered)} candidate chains from vector")
    return filtered