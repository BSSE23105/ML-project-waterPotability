# Keeps every prediction in a small SQLite database so past verdicts can be listed and summarised

import json
import os
import sqlite3
from collections import Counter
from datetime import datetime, timezone

SCHEMA = ("CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, "
          "source TEXT NOT NULL, inputs TEXT NOT NULL, prediction INTEGER NOT NULL, result TEXT NOT NULL, "
          "probability_potable REAL NOT NULL, confidence REAL NOT NULL, warnings TEXT NOT NULL, reasons TEXT NOT NULL)")
INSERT = ("INSERT INTO predictions (created_at, source, inputs, prediction, result, probability_potable, confidence, "
          "warnings, reasons) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)")


class PredictionHistory:
    """Append-only store of predictions with simple listing and statistics."""

    def __init__(self, db_path):
        self.db_path = db_path
        folder = os.path.dirname(db_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(SCHEMA)
        self.conn.commit()

    def add(self, result, source="api"):
        """Stores one prediction result and returns its new id."""
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cursor = self.conn.execute(INSERT, (
            stamp, source, json.dumps(result["input"]), result["prediction"], result["result"],
            result["probability_potable"], result["confidence"],
            json.dumps(result.get("warnings", [])), json.dumps(result.get("reasons", [])),
        ))
        self.conn.commit()
        return cursor.lastrowid

    def add_many(self, results, source="batch"):
        return [self.add(r, source) for r in results]

    @staticmethod
    def _row_to_dict(row):
        record = dict(row)
        for key in ("inputs", "warnings", "reasons"):
            record[key] = json.loads(record[key])
        return record

    def get(self, prediction_id):
        row = self.conn.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def list(self, limit=50, offset=0, result=None):
        """Newest first; optionally filtered to 'Potable' or 'Not Potable'."""
        where = " WHERE result = ?" if result else ""
        params = ([result] if result else []) + [limit, offset]
        rows = self.conn.execute(f"SELECT * FROM predictions{where} ORDER BY id DESC LIMIT ? OFFSET ?", params)
        return [self._row_to_dict(r) for r in rows.fetchall()]

    def count(self):
        return self.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    def stats(self):
        """Totals, unsafe share, average probability, commonest warnings and a per-day count."""
        rows = self.conn.execute("SELECT created_at, prediction, probability_potable, warnings FROM predictions").fetchall()
        total = len(rows)
        potable = sum(1 for r in rows if r["prediction"] == 1)
        warning_counter = Counter(w["feature"] for r in rows for w in json.loads(r["warnings"]))
        per_day = Counter(r["created_at"][:10] for r in rows)
        return {
            "total": total,
            "potable": potable,
            "not_potable": total - potable,
            "unsafe_share": round((total - potable) / total * 100, 2) if total else 0.0,
            "average_probability_potable": round(sum(r["probability_potable"] for r in rows) / total, 2) if total else 0.0,
            "most_common_warnings": [{"feature": f, "count": c} for f, c in warning_counter.most_common(5)],
            "predictions_per_day": [{"day": d, "count": c} for d, c in sorted(per_day.items())],
        }

    def clear(self):
        deleted = self.conn.execute("DELETE FROM predictions").rowcount
        self.conn.commit()
        return deleted

    def close(self):
        self.conn.close()
