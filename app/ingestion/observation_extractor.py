import math
from fastembed import TextEmbedding
from app.utils.embeddings import embed
from app.utils.logger import get_logger
import config

logger = get_logger("observation_extractor")

# Semantic activity categories — the extractor maps any description
# to the closest category by embedding similarity.
ACTIVITY_CATEGORIES = [
    "crouching or hiding near fence or perimeter",
    "walking or patrolling normally",
    "vehicle loitering or circling",
    "vehicle parked stationary",
    "running or moving rapidly",
    "climbing or attempting to breach perimeter",
    "standing and watching or surveilling",
    "group gathering near entrance",
    "delivery or authorized vehicle activity",
    "no significant activity",
    "security sweep completed all clear nothing found",
    "authorized security response to incident",
]

# Semantic entity categories — broad type classification.
ENTITY_CATEGORIES = [
    "person behaving suspiciously near perimeter",
    "authorized security personnel on patrol",
    "vehicle parked or stationary",
    "vehicle moving loitering or circling",
    "unknown object near restricted area",
    "unidentified individual",
    "delivery or authorized service vehicle",
    "maintenance or construction worker",
    "animal or wildlife",
]

_category_embeddings: dict = {}

def _get_category_embeddings(categories: list) -> list:
    key = tuple(categories)
    if key not in _category_embeddings:
        _category_embeddings[key] = [embed(c) for c in categories]
    return _category_embeddings[key]

def _cosine_sim(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def _classify(text: str, categories: list) -> tuple:
    text_emb  = embed(text)
    cat_embs  = _get_category_embeddings(categories)
    scores    = [_cosine_sim(text_emb, ce) for ce in cat_embs]
    best_idx  = scores.index(max(scores))
    return categories[best_idx], round(max(scores), 4), text_emb

def _extract_entity_fingerprint(description: str) -> str:
    desc = description.strip()
    if len(desc.split()) <= 8:
        return desc
    
    # Prioritize suspicious descriptors
    suspicious_keywords = ["hooded", "masked", "unknown", "unidentified", "with bag"]
    for kw in suspicious_keywords:
        if kw in desc.lower():
            idx = desc.lower().find(kw)
            return desc[:idx+len(kw)+15]  # take context around it
    
    # fallback to original logic
    words = desc.split()
    CONNECTORS = {"near", "at", "in", "on", "the", "a", "an", "now", "still", ...}
    phrase = []
    for w in words[:8]:
        if w.lower() in CONNECTORS and phrase:
            break
        phrase.append(w)
    return " ".join(phrase)


def time_to_minutes(time_str: str) -> int:
    try:
        h, m = time_str.split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return 0

def extract(frame: dict, telemetry: dict) -> dict:
    description = frame["description"]
    location    = telemetry.get("location", "Unknown")
    time_str    = telemetry.get("time", "00:00")

    activity, activity_conf, _ = _classify(description, ACTIVITY_CATEGORIES)
    entity_type, entity_conf, _ = _classify(description, ENTITY_CATEGORIES)

    # Specific entity fingerprint — more granular than entity_type
    entity_fingerprint = _extract_entity_fingerprint(description)

    embed_text = f"{description} {location} {activity}"
    embedding  = embed(embed_text)

    # Embed the fingerprint separately — used in chain matching
    fingerprint_embedding = embed(entity_fingerprint)

    obs = {
        "frame_id":              frame["frame_id"],
        "raw_description":       description,
        "activity":              activity,
        "activity_conf":         activity_conf,
        "entity_type":           entity_type,
        "entity_conf":           entity_conf,
        "entity_fingerprint":    entity_fingerprint,
        "fingerprint_embedding": fingerprint_embedding,
        "entities":              [entity_type],
        "location":              location,
        "time_str":              time_str,
        "time_minutes":          time_to_minutes(time_str),
        "altitude":              telemetry.get("altitude", 0),
        "drone_id":              telemetry.get("drone_id", "D-01"),
        "embedding":             embedding,
    }

    logger.debug(
        f"Frame {frame['frame_id']}: [{activity}] "
        f"entity=[{entity_type}] "
        f"fingerprint=[{entity_fingerprint}] "
        f"@ {location} conf=({activity_conf}, {entity_conf})"
    )
    return obs