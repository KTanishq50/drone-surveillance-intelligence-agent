import math
import config
from app.utils.embeddings import embed
from app.utils.logger import get_logger

logger = get_logger("event_synthesizer")

# ---------------------------------------------------------------------------
# Semantic descriptors
# ---------------------------------------------------------------------------

HIGH_RISK_DESCRIPTORS = [
    "masked individual crouching near restricted perimeter at night",
    "hooded person hiding near secure fence line",
    "person attempting perimeter breach",
    "unknown individual conducting surveillance of secure facility",
    "vehicle repeatedly circling restricted area",
    "person tampering with restricted gate or fence",
]

LOW_RISK_DESCRIPTORS = [
    "authorized security patrol",
    "delivery vehicle unloading at loading bay",
    "maintenance worker performing scheduled repair",
    "staff member entering facility normally",
    "empty area no suspicious activity",
    "security sweep completed area clear",
    "authorized security personnel responding to incident",
]

_risk_embeddings = None


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

def _get_risk_embeddings():
    global _risk_embeddings

    if _risk_embeddings is None:
        _risk_embeddings = {
            "high": [embed(x) for x in HIGH_RISK_DESCRIPTORS],
            "low": [embed(x) for x in LOW_RISK_DESCRIPTORS],
        }

    return _risk_embeddings


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def _cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))

    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))

    if na == 0 or nb == 0:
        return 0.0

    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Main semantic risk
# ---------------------------------------------------------------------------

def _semantic_risk_score(obs_embedding, observation):

    text = (
        observation.get("raw_description", "").lower()
        + " "
        + observation.get("activity", "").lower()
        + " "
        + " ".join(observation.get("entities", [])).lower()
    )

    # ---------------------------------------------------
    # CLEAR / SECURITY RESPONSE
    # ---------------------------------------------------

    clear_markers = [
        "all clear",
        "security sweep completed",
        "nothing found",
        "authorized security",
        "security guard",
        "patrol",
    ]

    if any(m in text for m in clear_markers):
        return 0.25

    # ---------------------------------------------------
    # Suspicious markers
    # ---------------------------------------------------

    suspicious_markers = {
        "hooded": 0.16,
        "masked": 0.18,
        "crouching": 0.20,
        "hiding": 0.20,
        "tampering": 0.22,
        "climbing": 0.24,
        "breach": 0.24,
        "surveilling": 0.18,
        "watching facility": 0.18,
        "unknown individual": 0.10,
        "concealed": 0.18,
    }

    score = 0.25

    for marker, weight in suspicious_markers.items():
        if marker in text:
            score += weight

    # ---------------------------------------------------
    # Semantic embedding contribution
    # ---------------------------------------------------

    embs = _get_risk_embeddings()

    high_scores = [
        _cosine_sim(obs_embedding, e)
        for e in embs["high"]
    ]

    low_scores = [
        _cosine_sim(obs_embedding, e)
        for e in embs["low"]
    ]

    high_max = max(high_scores)
    low_max = max(low_scores)

    semantic_delta = high_max - low_max

    if semantic_delta > 0.10:
        score += 0.15

    elif semantic_delta > 0.04:
        score += 0.08

    # ---------------------------------------------------
    # Vehicle logic
    # ---------------------------------------------------

    if "vehicle parked stationary" in text:
        score = min(score, 0.32)

    if "circling" in text or "loitering" in text:
        score = max(score, 0.48)

    return round(min(score, 0.92), 4)


# ---------------------------------------------------------------------------
# Contextual bonuses
# ---------------------------------------------------------------------------

def _night_bonus(time_minutes: int) -> float:
    is_night = (
        time_minutes >= config.NIGHT_START or
        time_minutes <= config.NIGHT_END
    )

    return 0.08 if is_night else 0.0


def _restricted_bonus(location: str) -> float:
    return 0.05 if location in config.RESTRICTED_LOCATIONS else 0.0


def _known_entity_bonus(entities: list, memory_hub) -> float:
    for entity in entities:

        if memory_hub.graph.node_exists(entity):
            relations = memory_hub.graph.get_ego_graph(entity, radius=1)

            for _, rel, _ in relations:
                if rel in (
                    "involved_in",
                    "flagged_as",
                    "escalated_to",
                ):
                    return 0.08

    return 0.0


# ---------------------------------------------------------------------------
# Main synthesis
# ---------------------------------------------------------------------------

def synthesize(observation: dict, memory_hub) -> dict:

    base_risk = _semantic_risk_score(
        observation["embedding"],
        observation,
    )

    text = (
        observation.get("raw_description", "").lower()
        + " "
        + observation.get("activity", "").lower()
    )

    # ---------------------------------------------------
    # no contextual inflation for benign scenes
    # ---------------------------------------------------

    if base_risk <= 0.32:
        bonus = 0.0

    else:
        bonus = (
            _night_bonus(observation["time_minutes"])
            + _restricted_bonus(observation["location"])
            + _known_entity_bonus(observation["entities"], memory_hub)
        )

    # ---------------------------------------------------
    # Security response suppressor
    # ---------------------------------------------------

    if any(m in text for m in [
        "all clear",
        "security sweep",
        "security guard",
        "authorized security",
    ]):
        bonus = 0.0
        base_risk = 0.25

    risk_score = round(min(base_risk + bonus, 1.0), 4)

    event_type = (
        observation["activity"]
        .replace(" ", "_")
        .replace(",", "")
    )

    embed_text = (
        f"{event_type} {observation['location']} "
        f"{observation['raw_description']}"
    )

    embedding = embed(embed_text)

    event = {
        "event_id": None,
        "frame_id": observation["frame_id"],
        "chain_id": None,
        "type": event_type,
        "entities": observation["entities"],
        "location": observation["location"],
        "time_str": observation["time_str"],
        "time_minutes": observation["time_minutes"],
        "risk_score": risk_score,
        "llm_risk": None,
        "embedding": embedding,
    }

    logger.debug(
        f"Synthesized event risk={risk_score} base={base_risk} bonus={bonus}"
    )

    return event