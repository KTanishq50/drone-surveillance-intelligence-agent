import math
import json
import config
from app.utils.embeddings import embed
from app.utils.logger import get_logger

logger = get_logger("context_assembler")

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _cosine_sim(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

# ---------------------------------------------------------------------------
# Entity category gating — broad person/vehicle separation
# ---------------------------------------------------------------------------

_PERSON_MARKERS  = {"person", "individual", "behaving", "suspicious",
                    "hooded", "masked", "unidentified", "authorized personnel",
                    "pedestrian", "figure", "intruder"}
_VEHICLE_MARKERS = {"vehicle", "stationary", "loitering",
                    "circling", "delivery truck", "pickup truck",
                    "sedan", "van", "car", "truck", "automobile"}

def _entity_category(entities: list) -> str:
    combined   = " ".join(entities).lower()
    is_person  = any(m in combined for m in _PERSON_MARKERS)
    is_vehicle = any(m in combined for m in _VEHICLE_MARKERS)
    if is_person and not is_vehicle:
        return "person"
    if is_vehicle and not is_person:
        return "vehicle"
    return "unknown"

def _categories_compatible(chain_entities: list, obs_entities: list) -> bool:
    chain_cat = _entity_category(chain_entities)
    obs_cat   = _entity_category(obs_entities)
    if chain_cat == "unknown" or obs_cat == "unknown":
        return True
    return chain_cat == obs_cat

# ---------------------------------------------------------------------------
# Fingerprint similarity — specific entity identity matching
# ---------------------------------------------------------------------------

_fp_embed_cache: dict = {}

def _fp_embed(text: str) -> list:
    if text not in _fp_embed_cache:
        _fp_embed_cache[text] = embed(text)
    return _fp_embed_cache[text]

def _fingerprint_similarity(chain: dict, observation: dict) -> float:
    """
    Compares the chain's stored entity fingerprint embedding against
    the current observation's fingerprint embedding.

    Returns a similarity score 0.0–1.0.
    A blue pickup truck observation scores ~0.85 against a blue pickup
    chain and ~0.55 against a silver sedan chain. This prevents different
    specific vehicles from accumulating into the same chain even when
    they share the broad entity type 'vehicle parked or stationary'.

    Falls back to 0.5 (neutral) when no fingerprint is available on
    either side — ensures legacy chains without fingerprints still work.
    """
    chain_fp_emb = chain.get("fingerprint_embedding", [])
    obs_fp_emb   = observation.get("fingerprint_embedding", [])

    if not chain_fp_emb or not obs_fp_emb:
        return 0.5  # neutral fallback — no fingerprint data

    return round(_cosine_sim(chain_fp_emb, obs_fp_emb), 4)

# ---------------------------------------------------------------------------
# Location matching
# ---------------------------------------------------------------------------

def _location_match(chain: dict, observation: dict) -> bool:
    chain_locs = [l.lower() for l in chain.get("locations", [])]
    obs_loc    = observation["location"].lower()
    adj_locs   = [
        l.lower() for l in
        config.LOCATION_ADJACENCY.get(observation["location"], [])
    ]
    return obs_loc in chain_locs or any(al in chain_locs for al in adj_locs)

# ---------------------------------------------------------------------------
# Chain match scoring
# ---------------------------------------------------------------------------

def _score_chain_match(
    chain: dict,
    observation: dict,
    vector_similarity: float
) -> float:
    """
    Four-signal score:
      0.35 — fingerprint similarity  (specific entity identity)
      0.30 — vector similarity       (ChromaDB semantic cosine)
      0.20 — location match          (same zone or adjacent)
      0.15 — entity type match       (broad person/vehicle category)
    """
    fp_sim = _fingerprint_similarity(chain, observation)
    fp_score = 0.35 * fp_sim

    # Broad entity type match — binary bonus
    entity_broad_match = _categories_compatible(
        chain.get("entities", []), observation.get("entities", [])
    )
    entity_score = 0.15 if entity_broad_match else 0.0

    location_score = 0.20 if _location_match(chain, observation) else 0.0
    vector_score   = 0.30 * vector_similarity

    return round(fp_score + entity_score + location_score + vector_score, 4)

# ---------------------------------------------------------------------------
# Candidate chain merging
# ---------------------------------------------------------------------------

def _merge_chain_candidates(
    vector_candidates: list,
    sql_chains: list
) -> list:
    merged: dict = {}
    for c in sql_chains:
        merged[c["chain_id"]] = {"chain": c, "vector_score": 0.0}
    for vc in vector_candidates:
        try:
            cid = int(vc["id"])
            if cid in merged:
                merged[cid]["vector_score"] = vc["similarity"]
        except (ValueError, KeyError):
            pass
    return [v for v in merged.values() if v["chain"] is not None]

# ---------------------------------------------------------------------------
# Risk recalculation
# ---------------------------------------------------------------------------

def _recalculate_risk(chain: dict, event: dict) -> float:
    chain_risk = chain["risk_score"]
    event_risk = event["risk_score"]
    disposition = chain.get("disposition", "neutral")

    # ---------------------------------------------------
    # BENIGN CHAINS
    # Completely frozen unless a clearly suspicious
    # event arrives.
    # ---------------------------------------------------
    if disposition == "benign":

        # genuinely suspicious event touching benign chain
        if event_risk >= 0.65:
            return round(max(chain_risk, 0.50), 4)

        # otherwise NEVER escalate
        return round(min(chain_risk, 0.32), 4)

    # ---------------------------------------------------
    # NORMAL ESCALATION
    # ---------------------------------------------------
    event_count = len(chain.get("event_ids", []))

    if event_risk <= 0.35:
        return round(max(chain_risk * 0.96, event_risk), 4)

    repeat_pressure = min(0.07 * event_count, 0.35)

    if disposition == "suspicious":
        repeat_pressure += 0.10

    if event_risk >= chain_risk:
        new_risk = (
            0.40 * chain_risk +
            0.60 * event_risk +
            repeat_pressure
        )
    else:
        blended = (
            0.65 * chain_risk +
            0.35 * event_risk
        )
        floored = chain_risk * 0.97
        new_risk = max(blended, floored) + repeat_pressure

    return round(min(new_risk, 1.0), 4)


# ---------------------------------------------------------------------------
# Disposition classification
# Determines if a chain is benign, neutral, or suspicious based on
# entity type and event risk. Called on chain creation and updated
# when new events arrive.
# ---------------------------------------------------------------------------

# Entity types that always start as benign regardless of risk score.
# These are authorised actors — the system knows who they are.
# -------------------------------------------------------------------
# Disposition & Risk Cap Logic
# -------------------------------------------------------------------

_BENIGN_ENTITY_TYPES = {
    "authorized security personnel on patrol",
    "delivery or authorized service vehicle",
    "maintenance or construction worker",
    "staff member",
    "security guard",
    "uniformed security",
}

_SUSPICIOUS_PROMOTION_THRESHOLD = 0.55

_BENIGN_RISK_CAP = 0.32   # Much lower cap for clean benign chains


def _is_benign_entity(observation: dict) -> bool:
    """Stronger check using both entity_type and fingerprint"""
    text = (
        observation.get("entity_fingerprint", "").lower() + " " +
        " ".join(observation.get("entities", [])).lower() +
        " " + observation.get("raw_description", "").lower()
    )
    return any(b in text for b in ["security", "guard", "staff", "delivery", "maintenance", "authorized"])


def _initial_disposition(observation: dict, event: dict) -> str:
    if _is_benign_entity(observation) and event["risk_score"] <= 0.45:
        return "benign"
    
    if event["risk_score"] >= _SUSPICIOUS_PROMOTION_THRESHOLD:
        return "suspicious"
    
    return "neutral"


def _update_disposition(chain: dict, event: dict) -> str:
    current = chain.get("disposition", "neutral")
    risk = event["risk_score"]

    if current == "suspicious":
        return "suspicious"

    if current == "benign":
        # Only promote benign chain if a clearly suspicious event arrives
        if risk >= 0.65:
            return "suspicious"
        if risk >= 0.52:
            return "neutral"
        return "benign"   # Stay benign unless real threat appears

    # neutral
    if risk >= _SUSPICIOUS_PROMOTION_THRESHOLD:
        return "suspicious"
    return "neutral"


def _apply_disposition_risk_cap(chain: dict) -> float:
    disposition = chain.get("disposition", "neutral")
    risk = chain["risk_score"]

    if disposition == "benign":
        return round(min(risk, _BENIGN_RISK_CAP), 4)   # ← This is the key change (0.32)
    if disposition == "neutral":
        return round(min(risk, 0.65), 4)
    return risk   # suspicious = no cap



def _is_security_response(observation: dict) -> bool:
    text = (
        observation.get("raw_description", "").lower()
        + " "
        + observation.get("activity", "").lower()
    )

    markers = [
        "security guard",
        "security response",
        "security sweep",
        "all clear",
        "authorized security",
        "patrol",
    ]

    return any(m in text for m in markers)

# ---------------------------------------------------------------------------
# Main assembler
# ---------------------------------------------------------------------------

def assemble_and_resolve_chain(
    observation: dict,
    worker_results: dict,
    memory_hub
) -> dict:
    retrieval = worker_results["retrieval"]
    synthesis = worker_results["synthesis"]
    chain_w   = worker_results["chain"]

    event          = synthesis["event"]
    all_sql_chains = chain_w["recent_chains_sql"]
    vector_chains  = retrieval["candidate_chains_vector"]

    merged = _merge_chain_candidates(vector_chains, all_sql_chains)

    best_chain       = None
    best_match_score = 0.0

    for entry in merged:
        chain = entry["chain"]

        # Hard gate: incompatible broad categories never link
        if not _categories_compatible(
            chain.get("entities", []), observation.get("entities", [])
        ):
            logger.debug(
                f"Chain {chain['chain_id']} skipped — category mismatch "
                f"(chain={_entity_category(chain.get('entities', []))} "
                f"obs={_entity_category(observation.get('entities', []))})"
            )
            continue

        # Floor gate: genuinely empty/benign observations don't link
        if event["risk_score"] <= 0.35:
            logger.debug(
                f"Chain {chain['chain_id']} skipped — "
                f"observation risk {event['risk_score']} at floor"
            )
            continue

        chain_is_suspicious = (
            entry["chain"].get("disposition") == "suspicious"
        )

        obs_is_security = _is_security_response(observation)

        # Prevent security responders from merging
        # into suspect behavior chains
        # ---------------------------------------------------
        # Security response isolation
        # Never merge security-response observations into
        # suspicious-looking perimeter/person behavior chains.
        # Security arriving at a scene is a NEW CONTEXT,
        # not continuation of the intruder behavior itself.
        # ---------------------------------------------------

        if obs_is_security:

            suspicious_markers = [
                "crouching",
                "hiding",
                "surveilling",
                "breach",
                "intruder",
                "masked",
                "hooded",
                "perimeter",
            ]

            chain_text = (
                chain.get("narrative", "").lower()
                + " "
                + " ".join(chain.get("entities", [])).lower()
            )

            if any(m in chain_text for m in suspicious_markers):
                logger.debug(
                    f"Chain {chain['chain_id']} skipped — "
                    f"security response isolated from suspicious chain"
                )
                continue

        score = _score_chain_match(chain, observation, entry["vector_score"])
        if score > best_match_score:
            best_match_score = score
            best_chain       = chain

    # -------------------------------------------------------------------
    # Link / reactivate / create
    # -------------------------------------------------------------------
    if best_chain and best_match_score >= config.CHAIN_LINK_THRESHOLD:
        was_dormant = best_chain["status"] == "dormant"

        if was_dormant:
            logger.info(
                f"Reactivating dormant chain {best_chain['chain_id']} "
                f"(match={best_match_score:.3f})"
            )
            best_chain["status"]       = "active"
            best_chain["_past_events"] = memory_hub.sql.get_events_for_chain(
                best_chain["chain_id"]
            )
        else:
            logger.info(
                f"Linking to active chain {best_chain['chain_id']} "
                f"(match={best_match_score:.3f})"
            )
            best_chain["_past_events"] = []

        event_ids = best_chain.get("event_ids", [])
        if observation["frame_id"] not in event_ids:
            event_ids.append(observation["frame_id"])
        best_chain["event_ids"] = event_ids

        # === DISPOSITION LOGIC ===
        best_chain["disposition"] = _update_disposition(best_chain, event)

        # Recalculate risk then apply disposition cap
        best_chain["risk_score"] = _recalculate_risk(best_chain, event)
        best_chain["risk_score"] = _apply_disposition_risk_cap(best_chain)

        if observation["location"] not in best_chain["locations"]:
            best_chain["locations"].append(observation["location"])
        for entity in observation["entities"]:
            if entity not in best_chain["entities"]:
                best_chain["entities"].append(entity)

        if observation.get("fingerprint_embedding"):
            best_chain["fingerprint_embedding"] = (
                observation["fingerprint_embedding"]
            )
        if observation.get("entity_fingerprint"):
            best_chain["entity_fingerprint"] = observation["entity_fingerprint"]

        resolved_chain = best_chain
        is_new_chain   = False

    else:
        logger.info(
            f"No matching chain (best={best_match_score:.3f} < "
            f"threshold={config.CHAIN_LINK_THRESHOLD}) — creating new chain"
        )
        narrative = (
            f"{observation['activity'].replace('_', ' ').replace(',', '')} "
            f"at {observation['location']}"
        )
        initial_disposition = _initial_disposition(observation, event)
        initial_risk = event["risk_score"]

        resolved_chain = {
            "chain_id":              None,
            "status":                "active",
            "disposition":           initial_disposition,
            "narrative":             narrative,
            "entities":              list(observation["entities"]),
            "locations":             [observation["location"]],
            "event_ids":             [],
            "risk_score":            initial_risk,
            "entity_fingerprint":    observation.get("entity_fingerprint", ""),
            "fingerprint_embedding": observation.get("fingerprint_embedding", []),
            "_past_events":          [],
        }
        # Apply cap immediately on creation
        resolved_chain["risk_score"] = _apply_disposition_risk_cap(resolved_chain)
        is_new_chain = True

    # -------------------------------------------------------------------
    # Chain embedding
    # -------------------------------------------------------------------
    chain_embed_text = (
        f"{resolved_chain['narrative']} "
        f"{' '.join(resolved_chain['entities'])} "
        f"{' '.join(resolved_chain['locations'])}"
    )
    resolved_chain["embedding"] = embed(chain_embed_text)

    # -------------------------------------------------------------------
    # Persist
    # -------------------------------------------------------------------
    event_id = memory_hub.sql.insert_event(event)
    event["event_id"] = event_id

    if is_new_chain:
        chain_id = memory_hub.sql.insert_chain(resolved_chain)
        resolved_chain["chain_id"] = chain_id
    else:
        memory_hub.sql.update_chain(resolved_chain)
        chain_id = resolved_chain["chain_id"]

    memory_hub.sql.update_event_chain(event_id, chain_id)
    event["chain_id"] = chain_id

    memory_hub.vector.upsert_chain(
        chain_id,
        resolved_chain["embedding"],
        {
            "status":    resolved_chain["status"],
            "narrative": resolved_chain["narrative"],
            "location":  observation["location"],
            "risk_score": str(resolved_chain["risk_score"]),
        }
    )

    memory_hub.vector.upsert_event(
        event_id, event["embedding"],
        {
            "type":      event["type"],
            "location":  event["location"],
            "time_str":  event["time_str"],
            "risk_score": str(event["risk_score"]),
        }
    )

    memory_hub.vector.upsert_frame(
        observation["frame_id"], observation["embedding"],
        {
            "description": observation["raw_description"],
            "location":    observation["location"],
            "time_str":    observation["time_str"],
            "activity":    observation["activity"],
        }
    )

    memory_hub.sql.insert_frame(observation)

    graph_triples = []
    for entity in observation["entities"]:
        graph_triples.append((entity, "involved_in",  f"chain_{chain_id}"))
        graph_triples.append((entity, "observed_at",  observation["location"]))
    graph_triples.append((f"event_{event_id}", "part_of",     f"chain_{chain_id}"))
    graph_triples.append((f"chain_{chain_id}", "observed_at", observation["location"]))
    memory_hub.graph.add_edges(graph_triples)
    memory_hub.graph.save()

    situation = {
        "observation":     observation,
        "event":           event,
        "chain":           resolved_chain,
        "is_new_chain":    is_new_chain,
        "similar_frames":  retrieval["similar_frames"],
        "similar_events":  retrieval["similar_events"],
        "graph_relations": retrieval["graph_relations"],
    }

    logger.info(
        f"Situation assembled — chain {chain_id} "
        f"risk={resolved_chain['risk_score']} "
        f"disposition={resolved_chain.get('disposition', 'neutral')} "
        f"new={is_new_chain} "
        f"match={best_match_score:.3f} "
        f"fp=[{observation.get('entity_fingerprint', '')}]"
    )
    return situation