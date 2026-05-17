import os
import logging
os.environ["ANONYMIZED_TELEMETRY"] = "false"
logging.getLogger("chromadb").setLevel(logging.ERROR)


import chromadb
from app.utils.logger import get_logger

logger = get_logger("vector_store")

class VectorStore:
    def __init__(self, chroma_path: str):
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.frames = self.client.get_or_create_collection("frames_collection")
        self.events = self.client.get_or_create_collection("events_collection")
        self.chains = self.client.get_or_create_collection("chains_collection")
        logger.info(f"ChromaDB initialised at {chroma_path}")

    def upsert_frame(self, frame_id: int, embedding: list, metadata: dict):
        self.frames.upsert(
            ids=[str(frame_id)],
            embeddings=[embedding],
            metadatas=[metadata]
        )

    def upsert_event(self, event_id: int, embedding: list, metadata: dict):
        self.events.upsert(
            ids=[str(event_id)],
            embeddings=[embedding],
            metadatas=[metadata]
        )

    def upsert_chain(self, chain_id: int, embedding: list, metadata: dict):
        self.chains.upsert(
            ids=[str(chain_id)],
            embeddings=[embedding],
            metadatas=[metadata]
        )

    def query_frames(self, embedding: list, top_k: int = 5) -> list:
        return self._query(self.frames, embedding, top_k)

    def query_events(self, embedding: list, top_k: int = 3) -> list:
        return self._query(self.events, embedding, top_k)

    def query_chains(self, embedding: list, top_k: int = 2) -> list:
        return self._query(self.chains, embedding, top_k)

    def _query(self, collection, embedding: list, top_k: int) -> list:
        try:
            count = collection.count()
            if count == 0:
                return []
            k = min(top_k, count)
            results = collection.query(
                query_embeddings=[embedding],
                n_results=k
            )
            out = []
            ids = results.get("ids", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            for i, doc_id in enumerate(ids):
                similarity = 1 - distances[i]
                out.append({
                    "id": doc_id,
                    "metadata": metas[i],
                    "similarity": round(similarity, 4)
                })
            return out
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return []