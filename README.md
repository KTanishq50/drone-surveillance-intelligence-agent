# Drone Surveillance Intelligence Agent

An AI-powered surveillance intelligence system that processes drone telemetry and camera observations to detect, correlate, reason about, and escalate suspicious activity in real time.

This project is not just a frame classifier or alert generator — it is a multi-layer memory-driven surveillance reasoning architecture that tracks evolving behavioral narratives across time using semantic embeddings, persistent chain memory, graph relationships, and selective LLM reasoning.

---

# Table of Contents

- [Overview](#overview)
- [Core Features](#core-features)
- [System Architecture](#system-architecture)
- [Intelligence Pipeline](#intelligence-pipeline)
- [Memory Architecture](#memory-architecture)
- [Chain-Based Behavioral Tracking](#chain-based-behavioral-tracking)
- [Dormant Chain Reactivation](#dormant-chain-reactivation)
- [Risk Scoring Engine](#risk-scoring-engine)
- [LLM Reasoning Layer](#llm-reasoning-layer)
- [Real-Time Dashboard](#real-time-dashboard)
- [VLM Video Processing Subsystem](#vlm-video-processing-subsystem)
- [Folder Structure](#folder-structure)
- [Data Flow](#data-flow)
- [Tech Stack](#tech-stack)
- [Why This Architecture Matters](#why-this-architecture-matters)
- [Example Threat Lifecycle](#example-threat-lifecycle)
- [API Endpoints](#api-endpoints)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Sample Workflow](#sample-workflow)
- [Engineering Decisions & Tradeoffs](#engineering-decisions--tradeoffs)
- [Future Improvements](#future-improvements)

---

# Demo Video

https://github.com/user-attachments/assets/f63e69ed-9c43-49b1-b9df-456db2907812

---

# Overview

The Drone Surveillance Intelligence Agent is a prototype AI surveillance system designed to simulate intelligent drone-based security monitoring.

The system processes:

- Drone camera frame descriptions
- Drone telemetry
- Behavioral observations
- Historical chain memory
- Graph relationships
- Semantic similarity retrieval

It then:

- Detects suspicious activity
- Tracks entities across time
- Reactivates dormant threat narratives
- Performs contextual risk escalation
- Calls an LLM only when necessary
- Generates actionable security alerts
- Streams live updates to a dashboard

Unlike traditional surveillance systems that treat every frame independently, this architecture models surveillance as evolving behavioral intelligence.

A suspicious event is not just an isolated observation.

It becomes part of a persistent behavioral chain.

---

# Core Features

## Semantic Surveillance Intelligence

Uses embedding similarity instead of brittle keyword matching.

Examples:
- “hooded man”
- “masked individual”
- “unknown figure near perimeter”

All map into the same semantic threat category.

---

## Behavioral Chain Tracking

Frames become:
- Observations
- Observations become events
- Events become evolving behavioral chains

The system remembers suspicious narratives over time.

---

## Dormant Threat Reactivation

If a suspicious entity reappears hours later:
- the old chain is reactivated
- prior context is restored
- escalation continues

The system does not forget prior suspicious behavior.

---

## Multi-Layer Memory Architecture

Three specialized memory systems:

| Memory Type | Purpose |
|---|---|
| SQLite | Ground-truth structured storage |
| ChromaDB | Semantic vector retrieval |
| NetworkX Graph | Relationship traversal |

---

## Risk-Gated LLM Reasoning

LLMs are expensive and unnecessary for benign frames.

The system:
- uses deterministic logic first
- calls the LLM only for elevated situations
- skips reasoning for low-risk events

---

## Real-Time Surveillance Dashboard

Live Server-Sent Events (SSE) dashboard with:
- streaming frame analysis
- alert visualization
- live chain state
- operator chat interface

---

## Vision-Language Model (VLM) Integration

Supports real video ingestion using:
- HuggingFace SmolVLM-256M-Instruct

The VLM subsystem:
- extracts frames from videos
- captions them
- generates `frames.json`
- feeds the main surveillance pipeline

---

# System Architecture

```text
┌────────────────────────────────────────────┐
│                ENTRY LAYER                 │
├────────────────────────────────────────────┤
│ CLI: python -m app.pipeline                │
│ FastAPI Web Server                         │
└────────────────────┬───────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│         OBSERVATION INGESTION              │
├────────────────────────────────────────────┤
│ frame_input.py                             │
│ telemetry_input.py                         │
│ observation_extractor.py                   │
└────────────────────┬───────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│       PARALLEL ASYNC WORKERS               │
├────────────────────────────────────────────┤
│ retrieval_worker                           │
│ synthesis_worker                           │
│ chain_worker                               │
└────────────────────┬───────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│         CONTEXT ASSEMBLER                  │
├────────────────────────────────────────────┤
│ Chain matching                             │
│ Risk escalation                            │
│ Dormant reactivation                       │
│ Persistence to all memory layers           │
└────────────────────┬───────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│              RISK GATE                     │
├────────────────────────────────────────────┤
│ risk >= 0.45 ?                             │
└───────────────┬────────────────────────────┘
                │ YES
                ▼
┌────────────────────────────────────────────┐
│             LLM REASONER                   │
├────────────────────────────────────────────┤
│ Groq Llama-3.1-8B-Instant                  │
└────────────────────┬───────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│             ALERT ENGINE                   │
├────────────────────────────────────────────┤
│ Alert generation                           │
│ Graph escalation                           │
│ SSE dashboard updates                      │
└────────────────────────────────────────────┘
```

---

# Intelligence Pipeline

## 1. Observation Extraction

Raw frame descriptions are transformed into structured observations.

Example:

```json
{
  "frame_id": 14,
  "raw_description": "Hooded man crouching near perimeter fence",
  "activity": "crouching or hiding near fence or perimeter",
  "entity_type": "person behaving suspiciously near perimeter",
  "location": "North Perimeter",
  "time_str": "22:47",
  "embedding": [...]
}
```

The extractor:
- embeds the text
- classifies activities semantically
- generates entity fingerprints
- attaches telemetry metadata

---

## 2. Parallel Worker Execution

The system runs three independent async workers simultaneously:

### Retrieval Worker
Fetches:
- similar frames
- similar events
- similar chains
- graph relations

### Synthesis Worker
Creates:
- event type
- risk score
- event embedding
- synthesized behavioral description

### Chain Worker
Fetches:
- active chains
- dormant chains
- reactivation candidates

All workers synchronize through:

```python
await asyncio.gather(...)
```

---

## 3. Context Assembly

The context assembler is the intelligence core.

It:
- merges worker outputs
- scores chain matches
- links or creates chains
- recalculates risk
- applies disposition logic
- persists all state

---

# Memory Architecture

## SQLite — Structured Ground Truth

Stores:
- frames
- events
- chains
- alerts

Acts as the authoritative historical record.

---

## ChromaDB — Semantic Memory

Supports:
- similarity search
- contextual retrieval
- dormant chain discovery
- semantic event correlation

Three collections:
- frames_collection
- events_collection
- chains_collection

---

## NetworkX Graph — Relationship Intelligence

Stores relationships between:
- entities
- chains
- events
- alerts
- locations

Enables:
- multi-hop reasoning
- escalation tracing
- contextual graph retrieval

Example relationships:

```text
entity ─ involved_in ─► chain
chain ─ escalated_to ─► alert
alert ─ triggered_by ─► event
```

---

# Chain-Based Behavioral Tracking

Traditional surveillance systems:
- treat events independently

This system:
- builds evolving behavioral narratives

Example:

```text
22:05 → Hooded figure near perimeter
22:28 → Same entity near restricted zone
23:10 → Fence tampering attempt
```

All become part of one evolving chain.

Risk escalates over time.

---

# Dormant Chain Reactivation

One of the most important architectural features.

If:
- a suspicious entity appears
- disappears for hours
- reappears later

The system:
- searches dormant chains
- compares embeddings
- compares entity fingerprints
- checks location adjacency
- restores prior chain context

This allows long-term behavioral intelligence.

---

# Risk Scoring Engine

The risk engine combines:

## Semantic Threat Similarity
Embedding comparison against:
- high-risk descriptors
- low-risk descriptors

---

## Keyword Suspicion Markers

Examples:
- hooded
- crouching
- breach
- tampering
- climbing

---

## Contextual Modifiers

Additional bonuses for:
- nighttime activity
- restricted locations
- prior suspicious graph relations

---

## Disposition System

Every chain becomes:

| Disposition | Meaning |
|---|---|
| benign | safe/authorized |
| neutral | uncertain |
| suspicious | confirmed threat |

Benign chains are capped and never escalate into alerts.

---

# LLM Reasoning Layer

The system uses:

- Groq API
- Llama-3.1-8B-Instant
- LangChain for prompt orchestration

LLM calls happen only if:

```python
risk >= 0.45
```

The LLM receives:
- observation context
- chain history
- graph relationships
- retrieved historical events

Expected structured output:

```json
{
  "suspiciousness": "high",
  "risk_level": 0.88,
  "reasoning": [
    "Repeated perimeter surveillance",
    "Dormant chain reactivation detected"
  ],
  "recommended_action": "Dispatch security"
}
```

---

# Real-Time Dashboard

The dashboard uses:
- FastAPI
- Server-Sent Events (SSE)
- Vanilla HTML/CSS/JS

Features:
- live frame stream
- animated event cards
- alert highlighting
- operator chat interface
- live statistics
- real-time reasoning display

No frontend framework.
No build step.

---

# VLM Video Processing Subsystem

The VLM subsystem converts real videos into surveillance observations.

Pipeline:

```text
Video Upload
    ↓
Frame Extraction
    ↓
SmolVLM Captioning
    ↓
frames.json Generation
    ↓
Main Surveillance Pipeline
```

Model used:

`HuggingFaceTB/SmolVLM-256M-Instruct`

Why this model:
- lightweight
- fast
- low VRAM requirements
- practical for prototype inference

---

# Folder Structure

```text
srvlnc_agent_clean/

├── app/
│   ├── alerts/
│   ├── engine/
│   ├── ingestion/
│   ├── memory/
│   ├── reasoning/
│   ├── retrieval/
│   ├── utils/
│   ├── workers/
│   ├── api.py
│   ├── bus.py
│   └── pipeline.py
│
├── data/
│   ├── db/
│   ├── graph/
│   ├── chroma/
│   └── simulation/
│
├── ui/
│   └── index.html
│
├── vlm/
│   ├── main.py
│   ├── uploads/
│   ├── outputs/
│   └── temp_frames/
│
├── config.py
├── requirements.txt
└── .env
```

---

# Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| API | FastAPI |
| Async Runtime | asyncio |
| Vector DB | ChromaDB |
| Structured Storage | SQLite |
| Graph Memory | NetworkX |
| Embeddings | fastembed |
| LLM | Groq Llama-3.1-8B |
| VLM | SmolVLM-256M |
| UI | Vanilla HTML/CSS/JS |

---

# Why This Architecture Matters

This project intentionally avoids:
- overengineering
- distributed infrastructure
- Kubernetes
- heavy orchestration frameworks
- unnecessary AI usage

Instead, it focuses on:
- correct architectural boundaries
- semantic intelligence
- memory layering
- persistent behavioral reasoning
- async coordination
- selective LLM usage

The goal is not scale.

The goal is intelligent system design.

---

# Example Threat Lifecycle

```text
Frame 5:
"Hooded figure near north perimeter"
→ new suspicious chain created

Frame 7:
"Same figure crouching near fence"
→ linked to existing chain
→ risk escalates

Frame 9:
"Fence tampering detected"
→ LLM reasoning triggered
→ alert generated

Next Day:
"Unknown hooded individual near perimeter"
→ dormant chain reactivated
→ escalation resumes
```

---

# API Endpoints

## Stream Endpoint

```http
GET /stream
```

Real-time SSE surveillance feed.

---

## Chat Endpoint

```http
POST /chat
```

Operator queries against surveillance memory.

Example:

```json
{
  "message": "Are there any active perimeter threats?"
}
```

---

## Dashboard

```http
GET /
```

Serves the real-time surveillance UI.

---

# Installation

## Clone Repository

```bash
git clone <repo-url>
cd srvlnc_agent_clean
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

Activate:

### Windows

```bash
venv\Scripts\activate
```

### Linux / WSL

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment

Create `.env`

```env
GROQ_API_KEY=your_api_key_here
```

---

# Running the Project

## CLI Mode

```bash
python -m app.pipeline
```

---

## Web Dashboard Mode

```bash
uvicorn app.api:app --reload --port 8000
```

Open:

```text
http://localhost:8000
```

---

# Sample Workflow

```text
1. Load frames + telemetry
2. Extract structured observations
3. Run async workers
4. Retrieve semantic history
5. Resolve behavioral chains
6. Escalate risk
7. Conditionally invoke LLM
8. Generate alerts
9. Stream results to dashboard
```

---

# Engineering Decisions & Tradeoffs

## Why ChromaDB Instead of FAISS?

ChromaDB provides:
- persistence
- metadata filtering
- simpler APIs
- less infrastructure overhead

Perfect for a prototype intelligence system.

---

## Why SQLite?

SQLite provides:
- zero-config persistence
- fast local access
- portability
- deterministic storage

Ideal for local AI systems.

---

## Why AsyncIO Instead of Queues/Brokers?

The workload is:
- local
- lightweight
- IO-bound

AsyncIO gives:
- concurrency
- simplicity
- minimal dependencies

Without introducing Kafka, Redis, or Celery complexity.

---

## Why Risk-Gated LLM Calls?

Most surveillance frames are benign.

Calling an LLM for every frame:
- wastes compute
- increases latency
- adds noise

The system uses deterministic intelligence first and escalates only when necessary.

---

# Future Improvements

## Potential Extensions

- Multi-drone coordination
- Live RTSP ingestion
- YOLO-based object detection
- Distributed processing
- Graph neural networks
- Long-term behavioral analytics
- Threat prediction
- Heatmap visualization
- Real-time GPS overlays
- Multi-agent orchestration
- Hybrid cloud deployment

---

# Final Notes

This project demonstrates:

- AI systems architecture
- semantic retrieval design
- persistent memory engineering
- asynchronous orchestration
- contextual reasoning
- graph-based intelligence
- selective LLM integration
- practical multimodal pipeline design

More importantly, it demonstrates an understanding that intelligent systems are not built by “adding an LLM everywhere.”

They are built by placing intelligence at the correct architectural layer.
