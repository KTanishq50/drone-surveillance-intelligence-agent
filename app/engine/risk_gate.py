import config
from app.utils.logger import get_logger

logger = get_logger("risk_gate")

def should_reason(situation: dict) -> bool:
    risk = situation["chain"]["risk_score"]
    result = risk >= config.LLM_RISK_THRESHOLD
    logger.info(f"Risk gate: score={risk} threshold={config.LLM_RISK_THRESHOLD} → {'PASS' if result else 'SKIP'}")
    return result