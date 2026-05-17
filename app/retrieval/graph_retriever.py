from app.utils.logger import get_logger

logger = get_logger("graph_retriever")

def retrieve(observation: dict, memory_hub) -> list:
    entities = observation.get("entities", [])
    triples  = memory_hub.graph.get_relations_for_entities(entities)
    logger.debug(f"Graph retrieval: {len(triples)} relation triples for {entities}")
    return triples