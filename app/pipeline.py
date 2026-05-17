import asyncio
import copy
import os

from app.memory.memory_hub import MemoryHub
from app.ingestion import frame_input, telemetry_input
from app.ingestion.observation_extractor import extract
from app.workers import retrieval_worker, synthesis_worker, chain_worker
from app.engine.context_assembler import assemble_and_resolve_chain
from app.engine.risk_gate import should_reason
from app.reasoning.llm_reasoner import reason
from app.alerts.alert_engine import process as process_alert
from app.utils.logger import get_logger
import config

logger = get_logger("pipeline")


# ---------------------------------------------------------------------------
# Single-frame processor — used by both CLI run() and API server
# ---------------------------------------------------------------------------

async def process_frame(observation: dict, memory_hub) -> dict:
    """
    Runs the three workers concurrently, assembles the situation,
    gates the LLM call, and runs the alert engine.
    Returns a complete result dict for the frame.
    """
    r_task = asyncio.create_task(
        retrieval_worker.run(copy.deepcopy(observation), memory_hub)
    )
    s_task = asyncio.create_task(
        synthesis_worker.run(copy.deepcopy(observation), memory_hub)
    )
    c_task = asyncio.create_task(
        chain_worker.run(copy.deepcopy(observation), memory_hub)
    )

    results = await asyncio.gather(r_task, s_task, c_task)
    worker_results = {r["worker"]: r for r in results}

    situation = assemble_and_resolve_chain(observation, worker_results, memory_hub)

    if should_reason(situation):
        llm_output = reason(situation)
    else:
        llm_output = {
            "suspiciousness":     "low",
            "risk_level":         situation["chain"]["risk_score"],
            "reasoning":          ["Below LLM threshold — rule-based score only"],
            "recommended_action": "Continue monitoring",
            "confidence":         "medium",
        }

    alert = process_alert(situation, llm_output, memory_hub)

    return {
        "frame_id":   observation["frame_id"],
        "situation":  situation,
        "llm_output": llm_output,
        "alert":      alert,
    }


# ---------------------------------------------------------------------------
# Startup helper — shared between CLI and API
# ---------------------------------------------------------------------------

def init_hub() -> MemoryHub:
    """
    Creates data directories, initialises MemoryHub,
    and marks stale chains dormant so they are available
    for reactivation on the current run.
    """
    os.makedirs("data/db",    exist_ok=True)
    os.makedirs("data/chroma", exist_ok=True)
    os.makedirs("data/graph",  exist_ok=True)

    hub = MemoryHub(config.DB_PATH, config.CHROMA_PATH, config.GRAPH_PATH)

    # Chains not updated in the last hour become dormant.
    # In a live system set this to your real patrol interval.
    # In simulation (re-running same frames) this ensures prior-run
    # chains are reactivation candidates rather than treated as live.
    hub.sql.mark_old_chains_dormant(max_age_seconds=3600)

    return hub


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def run():
    hub       = init_hub()
    frames    = frame_input.load_frames("data/simulation/frames.json")
    telemetry = telemetry_input.load_telemetry("data/simulation/telemetry.json")

    logger.info(f"Starting pipeline — {len(frames)} frames to process")

    for frame in frames:
        telem  = telemetry.get(frame["frame_id"], {})
        obs    = extract(frame, telem)
        result = await process_frame(obs, hub)

        alert = result["alert"]
        chain = result["situation"]["chain"]

        if alert:
            print(f"\n🚨 ALERT — Frame {result['frame_id']}")
            print(f"   Risk:       {alert['risk_level']}")
            print(f"   Level:      {alert['suspiciousness'].upper()}")
            print(f"   Action:     {alert['recommended_action']}")
            print(f"   Reason:     {alert['reasoning'][0]}")
            print(f"   Chain:      {chain['chain_id']} ({chain['status']})")
        else:
            print(
                f"   Frame {result['frame_id']} — "
                f"{obs['activity'][:45]} @ {obs['location']} — "
                f"risk={chain['risk_score']} — "
                f"chain={chain['chain_id']} ({chain['status']})"
            )


if __name__ == "__main__":
    asyncio.run(run())