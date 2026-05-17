import asyncio
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.memory.memory_hub import MemoryHub
from app.pipeline import process_frame, init_hub
from app.ingestion import frame_input, telemetry_input
from app.ingestion.observation_extractor import extract
from app.utils.logger import get_logger
import config

logger = get_logger("api")

event_queue: asyncio.Queue = asyncio.Queue()
hub: MemoryHub = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global hub
    hub = init_hub()
    asyncio.create_task(run_pipeline_loop())
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def run_pipeline_loop():
    frames    = frame_input.load_frames("data/simulation/frames.json")
    telemetry = telemetry_input.load_telemetry("data/simulation/telemetry.json")

    # Only process frames not already in the database.
    # This is the correct behavior for a continuous drone feed —
    # frame IDs are always incrementing and each is processed exactly once.
    already_processed = hub.sql.get_processed_frame_ids()
    new_frames = [f for f in frames if f["frame_id"] not in already_processed]

    if not new_frames:
        logger.info("No new frames to process — all frames already in database")
        await event_queue.put({"type": "done"})
        return

    logger.info(
        f"Pipeline loop starting — {len(new_frames)} new frames "
        f"({len(already_processed)} already processed)"
    )

    for frame in new_frames:
        telem  = telemetry.get(frame["frame_id"], {})
        obs    = extract(frame, telem)
        result = await process_frame(obs, hub)

        chain = result["situation"]["chain"]
        alert = result["alert"]
        llm   = result["llm_output"]

        payload = {
            "type":               "alert" if alert else "frame",
            "frame_id":           result["frame_id"],
            "description":        obs["raw_description"],
            "location":           obs["location"],
            "time_str":           obs["time_str"],
            "activity":           obs["activity"],
            "chain_id":           chain["chain_id"],
            "chain_status":       chain["status"],
            "risk_score":         chain["risk_score"],
            "suspiciousness":     llm.get("suspiciousness", "low"),
            "recommended_action": llm.get("recommended_action", ""),
            "reasoning":          llm.get("reasoning", []),
            "alert":              alert,
        }
        await event_queue.put(payload)
        await asyncio.sleep(1.5)

    await event_queue.put({"type": "done"})
    logger.info("Pipeline loop complete")


@app.get("/stream")
async def stream():
    async def event_generator():
        while True:
            try:
                event = await asyncio.wait_for(
                    event_queue.get(), timeout=30
                )
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "done":
                    break
            except asyncio.TimeoutError:
                yield 'data: {"type":"ping"}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Chat context builder
# ---------------------------------------------------------------------------

def _build_surveillance_context() -> str:
    """
    Builds a rich plaintext context for the chat LLM.
    Includes all chains regardless of status, recent alerts,
    raw frame descriptions per chain (so the LLM can match
    natural language like "blue truck" or "hooded man" to chains),
    and entity relationship triples from the graph.
    """
    import json as _json

    # All chains — active, dormant, escalated — sorted by risk
    cur = hub.sql.conn.execute(
        "SELECT * FROM chains ORDER BY risk_score DESC LIMIT 15"
    )
    chains = []
    for row in cur.fetchall():
        d = dict(row)
        d["entities"]  = _json.loads(d.get("entities")  or "[]")
        d["locations"] = _json.loads(d.get("locations") or "[]")
        d["event_ids"] = _json.loads(d.get("event_ids") or "[]")
        chains.append(d)

    alerts = hub.sql.get_recent_alerts(limit=10)

    chain_lines = []
    for c in chains:
        entities_str  = ", ".join(c["entities"])  or "unknown"
        locations_str = ", ".join(c["locations"]) or "unknown"

        # Pull raw frame descriptions for this chain so the operator
        # can ask about specific objects ("blue truck", "hooded man")
        # and the LLM can match them to the right chain.
        frame_descs = []
        if c["event_ids"]:
            placeholders = ",".join("?" * len(c["event_ids"]))
            rows = hub.sql.conn.execute(
                f"SELECT description FROM frames "
                f"WHERE frame_id IN ({placeholders}) "
                f"ORDER BY frame_id DESC LIMIT 3",
                c["event_ids"]
            ).fetchall()
            frame_descs = [r[0] for r in rows]

        desc_str = " | ".join(frame_descs) if frame_descs else "no frames yet"

        chain_lines.append(
            f"  Chain {c['chain_id']} [{c['status'].upper()}] "
            f"risk={c['risk_score']:.2f} | "
            f"narrative: {c['narrative']} | "
            f"entity types: {entities_str} | "
            f"locations: {locations_str} | "
            f"events: {len(c['event_ids'])} | "
            f"recent frame descriptions: {desc_str}"
        )

    alert_lines = []
    for a in alerts:
        reasoning    = _json.loads(a.get("reasoning") or "[]")
        first_reason = reasoning[0] if reasoning else "no details"
        alert_lines.append(
            f"  Alert {a['alert_id']} | chain {a['chain_id']} | "
            f"risk={a['risk_level']:.2f} | "
            f"suspicion={a['suspiciousness']} | "
            f"reason: {first_reason} | "
            f"action: {a['recommended_action']}"
        )

    graph_triples = []
    try:
        G = hub.graph.G
        for u, v, data in list(G.edges(data=True))[:20]:
            graph_triples.append(
                f"  {u} --{data.get('relation', 'linked')}→ {v}"
            )
    except Exception:
        pass

    parts = [
        f"TOTAL CHAINS: {len(chains)}  |  TOTAL ALERTS: {len(alerts)}\n"
    ]

    if chain_lines:
        parts.append(
            "ALL CHAINS (sorted by risk):\n" + "\n".join(chain_lines)
        )
    else:
        parts.append("ALL CHAINS: none yet")

    if alert_lines:
        parts.append("\nRECENT ALERTS:\n" + "\n".join(alert_lines))
    else:
        parts.append("\nRECENT ALERTS: none")

    if graph_triples:
        parts.append(
            "\nENTITY RELATIONSHIPS:\n" + "\n".join(graph_triples)
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------

CHAT_SYSTEM = """You are a drone surveillance intelligence assistant \
for a private property owner.
You have access to the complete current surveillance state shown below.
Answer the operator's questions in plain, direct language.
Reference specific chain IDs, risk scores, locations, and the exact \
wording from frame descriptions when relevant.
If the operator asks about a specific object or person by name \
(e.g. "blue truck", "hooded man"), look for that wording in the \
"recent frame descriptions" fields and report which chain it belongs to.
If something is genuinely not in the data, say so clearly.
Never fabricate chain IDs, locations, or risk scores.

CURRENT SURVEILLANCE STATE:
{context}"""


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(req: ChatRequest):
    context = _build_surveillance_context()

    from langchain_groq import ChatGroq
    from langchain.prompts import PromptTemplate
    from langchain.schema.output_parser import StrOutputParser

    llm = ChatGroq(
        model=config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        max_tokens=600,
        temperature=0.2,
    )
    prompt = PromptTemplate.from_template(
        CHAT_SYSTEM + "\n\nOperator question: {question}\n\nAnswer:"
    )
    chain_ = prompt | llm | StrOutputParser()

    try:
        answer = chain_.invoke({
            "context":  context,
            "question": req.message,
        })
    except Exception as e:
        logger.error(f"Chat LLM error: {e}")
        answer = f"Query processing error: {e}"

    return {
        "answer":         answer,
        "context_chains": context.count("Chain "),
        "context_alerts": context.count("Alert "),
    }


@app.get("/")
async def serve_ui():
    with open("ui/index.html") as f:
        return HTMLResponse(f.read())