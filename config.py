import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Paths ---
DB_PATH = "data/db/surveillance.db"
CHROMA_PATH = "data/chroma/"
GRAPH_PATH = "data/graph/memory.gpickle"

# --- Embedding ---
EMBED_MODEL = "BAAI/bge-small-en"

# --- LLM ---
GROQ_MODEL = "llama-3.1-8b-instant"
LLM_MAX_TOKENS = 512

# --- Risk Thresholds ---
LLM_RISK_THRESHOLD = 0.45
ALERT_THRESHOLD = 0.70
CHAIN_LINK_THRESHOLD = 0.60
VECTOR_SIMILARITY_FLOOR = 0.55

# --- Chain Behaviour ---
CHAIN_REACTIVATION_WINDOW = 86400  # 24 hours in seconds

# --- Risk Weights ---
RISK_WEIGHTS = {
    "night_time": 0.25,
    "restricted_zone": 0.20,
    "covert_activity": 0.20,
    "known_entity_flag": 0.15,
    "low_altitude": 0.10,
}

# --- Location Config ---
RESTRICTED_LOCATIONS = [
    "North Perimeter",
    "Garage Sector",
    "Server Room",
    "Gate Alpha",
]

LOCATION_ADJACENCY = {
    "North Perimeter": ["Gate Alpha", "East Fence"],
    "Garage Sector": ["Loading Bay", "South Exit"],
}

# --- Night Hours (minutes from 00:00) ---
NIGHT_START = 1200  # 20:00
NIGHT_END = 360     # 06:00

# --- Activity Types ---
COVERT_ACTIVITY_TYPES = [
    "crouching_near_perimeter",
    "covert_approach_restricted_perimeter",
    "perimeter_surveillance",
    "fence_inspection",
    "loitering_restricted",
    "vehicle_loitering",
    "vehicle_surveillance",
]

# Entity embedding similarity threshold for chain matching
ENTITY_MATCH_THRESHOLD = 0.72