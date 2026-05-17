from app.utils.logger import get_logger, log_json
import config

logger = get_logger("alert_engine")

def process(situation: dict, llm_output: dict, memory_hub) -> dict | None:
    risk_level = llm_output.get("risk_level", 0.0)
    chain      = situation["chain"]
    event      = situation["event"]
    obs        = situation["observation"]

    memory_hub.sql.update_event_llm_risk(event["event_id"], risk_level)

    if risk_level < config.ALERT_THRESHOLD:
        logger.info(
            f"No alert — risk {risk_level} below threshold {config.ALERT_THRESHOLD}"
        )
        return None

    alert = {
        "chain_id":          chain["chain_id"],
        "frame_id":          obs["frame_id"],
        "event_id":          event["event_id"],
        "risk_level":        risk_level,
        "suspiciousness":    llm_output.get("suspiciousness", "unknown"),
        "reasoning":         llm_output.get("reasoning", []),
        "recommended_action": llm_output.get("recommended_action", "Review"),
    }

    alert_id = memory_hub.sql.insert_alert(alert)
    alert["alert_id"] = alert_id

    chain["status"]     = "escalated"
    chain["risk_score"] = risk_level
    memory_hub.sql.update_chain(chain)

    memory_hub.graph.add_edges([
        (f"chain_{chain['chain_id']}", "escalated_to", f"alert_{alert_id}"),
        (f"alert_{alert_id}", "triggered_by", f"event_{event['event_id']}"),
    ])
    memory_hub.graph.save()

    log_json(logger, f"ALERT GENERATED [id={alert_id}]", alert)
    return alert