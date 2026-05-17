import os
from app.memory.sqlite_store import SQLiteStore
from app.memory.vector_store import VectorStore
from app.memory.graph_store import GraphStore
from app.utils.logger import get_logger

logger = get_logger("memory_hub")

class MemoryHub:
    def __init__(self, db_path: str, chroma_path: str, graph_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(chroma_path, exist_ok=True)
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)

        self.sql = SQLiteStore(db_path)
        self.vector = VectorStore(chroma_path)
        self.graph = GraphStore(graph_path)
        logger.info("MemoryHub ready")