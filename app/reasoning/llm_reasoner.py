import json
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from app.utils.logger import get_logger
import config

logger = get_logger("llm_reasoner")

PROMPT_TEMPLATE = """You are a drone surveillance threat analyst.
Analyse the situation below and respond ONLY with a valid JSON object.
Do not include any text, explanation, or markdown outside the JSON.

CURRENT OBSERVATION:
- Description: {description}
- Location: {location}
- Time: {time_str}
- Activity type: {activity}
- Entities detected: {entities}

BEHAVIOURAL CHAIN:
- Narrative: {chain_narrative}
- Status: {chain_status}
- Events in chain: {chain_event_count}
- Chain risk score: {chain_risk}

HISTORICAL CONTEXT:
- Similar past frames: {similar_frames}
- Similar past events: {similar_events}
- Entity relationships: {graph_relations}

Respond with this exact JSON structure:
{{
  "suspiciousness": "low|medium|high|critical",
  "risk_level": <float 0.0 to 1.0>,
  "reasoning": ["<point 1>", "<point 2>", "<point 3>"],
  "recommended_action": "<single actionable instruction>",
  "confidence": "low|medium|high"
}}"""

def _format_frames(frames: list) -> str:
    if not frames:
        return "none"
    return "; ".join(
        f"{f['metadata'].get('description', '')} @ {f['metadata'].get('location', '')} (sim={f['similarity']})"
        for f in frames[:2]
    )

def _format_events(events: list) -> str:
    if not events:
        return "none"
    return "; ".join(
        f"{e['metadata'].get('type', '')} @ {e['metadata'].get('location', '')} (sim={e['similarity']})"
        for e in events[:2]
    )

def _format_relations(relations: list) -> str:
    if not relations:
        return "none"
    return "; ".join(f"{s} --{r}--> {t}" for s, r, t in relations[:5])

def _parse_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    return json.loads(text)

def reason(situation: dict) -> dict:
    obs   = situation["observation"]
    chain = situation["chain"]

    llm   = ChatGroq(
        model=config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        max_tokens=config.LLM_MAX_TOKENS,
        temperature=0.2,
    )
    prompt  = PromptTemplate.from_template(PROMPT_TEMPLATE)
    chain_  = prompt | llm | StrOutputParser()

    input_vars = {
        "description":       obs["raw_description"],
        "location":          obs["location"],
        "time_str":          obs["time_str"],
        "activity":          obs["activity"],
        "entities":          ", ".join(obs.get("entities", [])),
        "chain_narrative":   chain.get("narrative", "N/A"),
        "chain_status":      chain.get("status", "active"),
        "chain_event_count": len(chain.get("event_ids", [])),
        "chain_risk":        chain.get("risk_score", 0.0),
        "similar_frames":    _format_frames(situation.get("similar_frames", [])),
        "similar_events":    _format_events(situation.get("similar_events", [])),
        "graph_relations":   _format_relations(situation.get("graph_relations", [])),
    }

    try:
        raw = chain_.invoke(input_vars)
        result = _parse_response(raw)
        logger.info(f"LLM reasoning complete: {result.get('suspiciousness')} / {result.get('risk_level')}")
        return result
    except Exception as e:
        logger.error(f"LLM reasoning failed: {e} — using fallback")
        return {
            "suspiciousness":     "medium",
            "risk_level":         chain.get("risk_score", 0.5),
            "reasoning":          ["LLM unavailable — rule-based risk used"],
            "recommended_action": "Review manually",
            "confidence":         "low",
        }