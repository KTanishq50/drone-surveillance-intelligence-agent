import sqlite3
import json
from datetime import datetime, timedelta
from app.utils.logger import get_logger

logger = get_logger("sqlite_store")


class SQLiteStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()
        logger.info(f"SQLite initialised at {db_path}")

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS frames (
                frame_id              INTEGER PRIMARY KEY,
                drone_id              TEXT,
                description           TEXT,
                location              TEXT,
                time_str              TEXT,
                time_minutes          INTEGER,
                altitude              REAL,
                activity              TEXT,
                risk_score            REAL,
                created_at            TEXT
            );

            CREATE TABLE IF NOT EXISTS events (
                event_id              INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id              INTEGER,
                chain_id              INTEGER,
                type                  TEXT,
                entities              TEXT,
                location              TEXT,
                time_str              TEXT,
                time_minutes          INTEGER,
                risk_score            REAL,
                llm_risk              REAL,
                created_at            TEXT
            );

            CREATE TABLE IF NOT EXISTS chains (
                chain_id              INTEGER PRIMARY KEY AUTOINCREMENT,
                status                TEXT,
                disposition           TEXT,
                narrative             TEXT,
                entities              TEXT,
                locations             TEXT,
                event_ids             TEXT,
                risk_score            REAL,
                entity_fingerprint    TEXT,
                fingerprint_embedding TEXT,
                created_at            TEXT,
                last_updated          TEXT
            );

            CREATE TABLE IF NOT EXISTS alerts (
                alert_id              INTEGER PRIMARY KEY AUTOINCREMENT,
                chain_id              INTEGER,
                frame_id              INTEGER,
                event_id              INTEGER,
                risk_level            REAL,
                suspiciousness        TEXT,
                reasoning             TEXT,
                recommended_action    TEXT,
                timestamp             TEXT
            );
        """)
        existing = {
            row[1] for row in
            self.conn.execute("PRAGMA table_info(chains)").fetchall()
        }
        for col, default in [
            ("entity_fingerprint",    "''"),
            ("fingerprint_embedding", "''"),
            ("disposition",           "'neutral'"),
        ]:
            if col not in existing:
                self.conn.execute(
                    f"ALTER TABLE chains ADD COLUMN {col} TEXT DEFAULT {default}"
                )
        self.conn.commit()
    

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------

    def insert_frame(self, obs: dict) -> int:
        self.conn.execute("""
            INSERT OR REPLACE INTO frames
            (frame_id, drone_id, description, location, time_str,
             time_minutes, altitude, activity, risk_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            obs["frame_id"],
            obs.get("drone_id"),
            obs["raw_description"],
            obs["location"],
            obs["time_str"],
            obs["time_minutes"],
            obs.get("altitude"),
            obs.get("activity"),
            obs.get("risk_score", 0.0),
            datetime.now().isoformat(),
        ))
        self.conn.commit()
        return obs["frame_id"]

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def insert_event(self, event: dict) -> int:
        cur = self.conn.execute("""
            INSERT INTO events
            (frame_id, chain_id, type, entities, location, time_str,
             time_minutes, risk_score, llm_risk, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event["frame_id"],
            event.get("chain_id"),
            event["type"],
            json.dumps(event.get("entities", [])),
            event["location"],
            event["time_str"],
            event["time_minutes"],
            event.get("risk_score", 0.0),
            event.get("llm_risk"),
            datetime.now().isoformat(),
        ))
        self.conn.commit()
        return cur.lastrowid

    def get_events_for_chain(self, chain_id: int) -> list:
        cur = self.conn.execute(
            "SELECT * FROM events WHERE chain_id = ? ORDER BY created_at ASC",
            (chain_id,)
        )
        return [dict(row) for row in cur.fetchall()]

    def update_event_chain(self, event_id: int, chain_id: int):
        self.conn.execute(
            "UPDATE events SET chain_id = ? WHERE event_id = ?",
            (chain_id, event_id)
        )
        self.conn.commit()

    def update_event_llm_risk(self, event_id: int, llm_risk: float):
        self.conn.execute(
            "UPDATE events SET llm_risk = ? WHERE event_id = ?",
            (llm_risk, event_id)
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Chains
    # ------------------------------------------------------------------

    def _deserialise_chain(self, row) -> dict:
        import json as _json
        d = dict(row)
        d["entities"]  = _json.loads(d.get("entities")  or "[]")
        d["locations"] = _json.loads(d.get("locations") or "[]")
        d["event_ids"] = _json.loads(d.get("event_ids") or "[]")
        raw_fp = d.get("fingerprint_embedding") or "[]"
        try:
            d["fingerprint_embedding"] = _json.loads(raw_fp)
        except Exception:
            d["fingerprint_embedding"] = []
        if not d.get("disposition"):
            d["disposition"] = "neutral"
        return d

    def insert_chain(self, chain: dict) -> int:
        import json as _json
        now    = datetime.now().isoformat()
        fp_emb = _json.dumps(chain.get("fingerprint_embedding") or [])
        cur = self.conn.execute("""
            INSERT INTO chains
            (status, disposition, narrative, entities, locations, event_ids,
            risk_score, entity_fingerprint, fingerprint_embedding,
            created_at, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chain["status"],
            chain.get("disposition", "neutral"),
            chain["narrative"],
            _json.dumps(chain.get("entities", [])),
            _json.dumps(chain.get("locations", [])),
            _json.dumps(chain.get("event_ids", [])),
            chain.get("risk_score", 0.0),
            chain.get("entity_fingerprint", ""),
            fp_emb,
            now, now,
        ))
        self.conn.commit()
        return cur.lastrowid

    def update_chain(self, chain: dict):
        import json as _json
        fp_emb = _json.dumps(chain.get("fingerprint_embedding") or [])
        self.conn.execute("""
            UPDATE chains SET
                status                = ?,
                disposition           = ?,
                narrative             = ?,
                entities              = ?,
                locations             = ?,
                event_ids             = ?,
                risk_score            = ?,
                entity_fingerprint    = ?,
                fingerprint_embedding = ?,
                last_updated          = ?
            WHERE chain_id = ?
        """, (
            chain["status"],
            chain.get("disposition", "neutral"),
            chain["narrative"],
            _json.dumps(chain.get("entities", [])),
            _json.dumps(chain.get("locations", [])),
            _json.dumps(chain.get("event_ids", [])),
            chain["risk_score"],
            chain.get("entity_fingerprint", ""),
            fp_emb,
            datetime.now().isoformat(),
            chain["chain_id"],
        ))
        self.conn.commit()

    def get_chain(self, chain_id: int) -> dict | None:
        cur = self.conn.execute(
            "SELECT * FROM chains WHERE chain_id = ?", (chain_id,)
        )
        row = cur.fetchone()
        return self._deserialise_chain(row) if row else None

    def get_active_and_dormant_chains(self) -> list:
        """
        Returns active, dormant, AND escalated chains as linking candidates.
        Escalated chains can receive new events — a new observation of the
        same entity at the same location should join the existing chain,
        not create a new one. The chain manager will update status accordingly.
        """
        cur = self.conn.execute("""
            SELECT * FROM chains
            WHERE status IN ('active', 'dormant', 'escalated')
            ORDER BY last_updated DESC
            LIMIT 20
        """)
        return [self._deserialise_chain(row) for row in cur.fetchall()]

    def get_dormant_chains_within_window(self, window_seconds: int) -> list:
        cutoff = (
            datetime.now() - timedelta(seconds=window_seconds)
        ).isoformat()
        cur = self.conn.execute("""
            SELECT * FROM chains
            WHERE status = 'dormant' AND last_updated > ?
            ORDER BY last_updated DESC
        """, (cutoff,))
        return [self._deserialise_chain(row) for row in cur.fetchall()]

    def mark_old_chains_dormant(self, max_age_seconds: int):
        """
        Any chain still marked 'active' that has not been updated
        within max_age_seconds is transitioned to 'dormant'.
        Called once at pipeline startup so prior-session chains
        are available for reactivation rather than treated as live.
        """
        cutoff = (
            datetime.now() - timedelta(seconds=max_age_seconds)
        ).isoformat()
        self.conn.execute("""
            UPDATE chains
            SET    status = 'dormant'
            WHERE  status = 'active'
            AND    last_updated < ?
        """, (cutoff,))
        self.conn.commit()
        n = self.conn.execute("SELECT changes()").fetchone()[0]
        if n:
            logger.info(
                f"Marked {n} chain(s) dormant "
                f"(inactive > {max_age_seconds}s)"
            )

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def insert_alert(self, alert: dict) -> int:
        cur = self.conn.execute("""
            INSERT INTO alerts
            (chain_id, frame_id, event_id, risk_level, suspiciousness,
             reasoning, recommended_action, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.get("chain_id"),
            alert.get("frame_id"),
            alert.get("event_id"),
            alert["risk_level"],
            alert["suspiciousness"],
            json.dumps(alert.get("reasoning", [])),
            alert["recommended_action"],
            datetime.now().isoformat(),
        ))
        self.conn.commit()
        return cur.lastrowid

    def get_alerts_for_chain(self, chain_id: int) -> list:
        cur = self.conn.execute(
            "SELECT * FROM alerts WHERE chain_id = ? ORDER BY timestamp DESC",
            (chain_id,)
        )
        return [dict(row) for row in cur.fetchall()]

    def get_recent_alerts(self, limit: int = 10) -> list:
        cur = self.conn.execute(
            "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        return [dict(row) for row in cur.fetchall()]
    





    def get_processed_frame_ids(self) -> set:
        """Returns the set of all frame_ids already in the database."""
        cur = self.conn.execute("SELECT frame_id FROM frames")
        return {row[0] for row in cur.fetchall()}