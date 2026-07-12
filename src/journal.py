"""
Platform Architecture: Operational Ledger & Journal Database
Environment: Conda fx39 (Python 3.9 win-64)
Dependencies: sqlite3 (Native standard library)
"""

import sqlite3
import os
from typing import Dict, Any

class TradingJournal:
    """Uses localized zero-config SQLite database engines to cache execution telemetries."""
    
    def __init__(self, db_path: str = "logs/trading_state.db"):
        self.db_path = db_path
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Sets up internal tracking tables for active telemetry streaming."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table to track executions and system rejections
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS journal_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT,  -- 'TRADE_EXECUTION', 'TRADE_REJECTED', 'SYSTEM_ABORT'
                symbol TEXT,
                action TEXT,
                price REAL,
                lots REAL,
                reason TEXT
            )
        """)
        conn.commit()
        conn.close()

    def log_event(self, event_type: str, symbol: str, action: str, price: float, lots: float, reason: str) -> None:
        """Asynchronously writes operational events into the permanent ledger file."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO journal_logs (event_type, symbol, action, price, lots, reason)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (event_type, symbol, action, price, lots, reason))
        conn.commit()
        conn.close()