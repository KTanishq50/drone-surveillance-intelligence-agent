from app.utils.embeddings import embed
import config

def filter_by_threshold(results: list, floor: float = None) -> list:
    floor = floor or config.VECTOR_SIMILARITY_FLOOR
    return [r for r in results if r["similarity"] >= floor]