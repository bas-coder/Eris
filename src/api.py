"""
Platform Architecture: Supervisor API (FastAPI)
Environment: Conda fx39 (Python 3.9 win-64)
Dependencies: fastapi, uvicorn, pandas, sqlite3
"""

import sqlite3
import os
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="Ruthless Student Command Center")

# Allow cross-origin requests for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "trading_state.db")

@app.get("/api/logs")
def get_recent_logs():
    """Fetches the most recent 100 neural decisions and trade executions."""
    if not os.path.exists(DB_PATH):
        return {"error": "Database not initialized yet. Run main.py first."}
        
    try:
        conn = sqlite3.connect(DB_PATH)
        # Pull the latest 50 logs from the SQLite ledger
        query = "SELECT * FROM journal_logs ORDER BY timestamp DESC LIMIT 50"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert DataFrame to a list of dictionaries for JSON consumption
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/stats")
def get_bot_stats():
    """Calculates win rate and equity metrics (Mocked for dashboard layout verification)."""
    # Once we have real equity tracking, we will compute this from the DB.
    return {
        "win_rate": "58.4%",
        "profit_factor": "1.82",
        "total_trades": 142,
        "active_regime": "London Breakout (High Volatility)"
    }

@app.get("/")
def serve_dashboard():
    """Serves the React frontend application."""
    html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public", "index.html")
    return FileResponse(html_path)