from app.utils.logger import get_logger
import config

logger = get_logger("chain_worker")

async def run(observation: dict, memory_hub) -> dict:
    logger.debug(f"Chain worker starting for frame {observation['frame_id']}")

    recent_chains  = memory_hub.sql.get_active_and_dormant_chains()
    dormant_chains = memory_hub.sql.get_dormant_chains_within_window(
        config.CHAIN_REACTIVATION_WINDOW
    )

    return {
        "frame_id":             observation["frame_id"],
        "worker":               "chain",
        "recent_chains_sql":    recent_chains,
        "dormant_candidates_sql": dormant_chains,
    }