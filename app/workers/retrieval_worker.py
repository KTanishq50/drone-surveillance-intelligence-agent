from app.retrieval import (
    frame_retriever,
    event_retriever,
    chain_retriever,
    graph_retriever,
)
from app.utils.logger import get_logger

logger = get_logger("retrieval_worker")

async def run(observation: dict, memory_hub) -> dict:
    logger.debug(f"Retrieval worker starting for frame {observation['frame_id']}")

    similar_frames   = frame_retriever.retrieve(observation, memory_hub)
    similar_events   = event_retriever.retrieve(observation, memory_hub)
    candidate_chains = chain_retriever.retrieve(observation, memory_hub)
    graph_relations  = graph_retriever.retrieve(observation, memory_hub)

    return {
        "frame_id":        observation["frame_id"],
        "worker":          "retrieval",
        "similar_frames":  similar_frames,
        "similar_events":  similar_events,
        "candidate_chains_vector": candidate_chains,
        "graph_relations": graph_relations,
    }