"""
Platform Architecture: Right Brain Data Layer & Custom Gym Environment
Environment: Conda fx39 (Python 3.9 win-64)
Dependencies: gymnasium, numpy, pandas, databento, certifi
"""

import os
import ssl
from typing import Tuple, Dict, Any, Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

# =====================================================================
# WINDOWS SSL CORRUPTION MONKEY-PATCH
# =====================================================================
# This patch intercepts Python's default SSL context establishment. 
# If your local Windows certificate store throws an ASN1 parsing error 
# (e.g., [ASN1: NOT_ENOUGH_DATA]), this bypasses the bad registry entry 
# and defaults to the standard 'certifi' security bundle to prevent crashes.
# =====================================================================
_orig_create_default_context = ssl.create_default_context

def _safe_create_default_context(purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None):
    try:
        return _orig_create_default_context(purpose=purpose, cafile=cafile, capath=capath, cadata=cadata)
    except ssl.SSLError:
        print("[SYSTEM] Bypassing corrupted Windows SSL Store to keep bot alive...")
        context = ssl.SSLContext(purpose)
        try:
            import certifi
            context.load_verify_locations(cafile=certifi.where())
        except ImportError:
            # Absolute fallback if certifi is not locally present
            context.verify_mode = ssl.CERT_NONE
        return context

ssl.create_default_context = _safe_create_default_context
# =====================================================================

try:
    import databento as db
except ImportError:
    db = None


def fetch_historical_ticks(api_key: str, start_date: str, end_date: str, symbol: str = "6EU4") -> pd.DataFrame:
    """
    Fetches market-by-price (MBP-1) top-of-book tick data using Databento.
    Processes historical CME futures ticks to generate real-world time-series.
    """
    if db is None:
        raise ImportError("The 'databento' library is missing from your fx39 environment.")
        
    client = db.Historical(api_key)
    
    # Pulling high-resolution data from CME Globex matching the top-of-book schema
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3", 
        schema="mbp-1", 
        symbols=[symbol],
        start=start_date, 
        end=end_date,
    )
    
    df = data.to_df()
    
    # Safely map columns from MBP-1 schema into expected state array attributes
    if 'ask_px_00' in df.columns and 'bid_px_00' in df.columns:
        df['price'] = (df['ask_px_00'] + df['bid_px_00']) / 2.0
        df['spread'] = df['ask_px_00'] - df['bid_px_00']
    else:
        # Fallback mechanism if schema varies across custom contracts
        df['price'] = df['price'] if 'price' in df.columns else 0.0
        df['spread'] = 0.0001  # Standard EUR/USD default spread approximation (1 pip)
        
    return df


class EurUsdScalpingEnv(gym.Env):
    """
    Custom Multi-Modal Trading Environment for EUR/USD Scalping.
    Translates raw price, VWAP divergence, spread structures, 
    and macroeconomic news metrics into a stable RL state space.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, historical_data: pd.DataFrame):
        super(EurUsdScalpingEnv, self).__init__()
        
        self.data = historical_data.reset_index(drop=True)
        self.total_steps = len(self.data)
        self.current_step = 0
        
        # Verify and calculate fallback metrics if they are not in the raw data frame
        if 'vwap' not in self.data.columns:
            self._calculate_baseline_vwap()
            
        if 'sentiment' not in self.data.columns:
            self.data['sentiment'] = 0.0 # Neural base value
            
        if 'spread' not in self.data.columns:
            self.data['spread'] = 0.0001

        # Action Space: 0 = Hold/Flat, 1 = Buy (Fade High), 2 = Sell (Fade Low)
        self.action_space = spaces.Discrete(3)
        
        # Observation Space Array [Price, VWAP_Deviation, Spread, Sentiment]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(4,), 
            dtype=np.float32
        )

    def _calculate_baseline_vwap(self) -> None:
        """Fallback internal mathematical calculation for session VWAP."""
        typical_price = self.data['price']
        volume = self.data['size'] if 'size' in self.data.columns else 1.0
        
        cumulative_tp_v = (typical_price * volume).cumsum()
        cumulative_v = volume.cumsum()
        
        # Standard zero division guard
        cumulative_v = np.where(cumulative_v == 0, 1, cumulative_v)
        self.data['vwap'] = cumulative_tp_v / cumulative_v

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment sequence back to the first market data frame row."""
        super().reset(seed=seed)
        self.current_step = 0
        
        info = {"status": "Environment Reset Successful"}
        return self._get_obs(), info

    def _get_obs(self) -> np.ndarray:
        """Constructs a flat state observation matrix for policy evaluation."""
        row = self.data.iloc[self.current_step]
        
        price = float(row['price'])
        vwap_val = float(row['vwap'])
        vwap_deviation = price - vwap_val
        spread = float(row['spread'])
        sentiment = float(row['sentiment'])
        
        return np.array([price, vwap_deviation, spread, sentiment], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one tick update step through our offline dataset sequence."""
        self.current_step += 1
        
        # Trace if we have fully exhausted the tick dataset sequence limits
        terminated = self.current_step >= (self.total_steps - 1)
        truncated = False
        
        obs = self._get_obs()
        info = {"tick_index": self.current_step}
        
        reward = 0.0
        vwap_dev = obs[1]
        
        # Evaluation Logic: Positive rewards are assigned if the bot's position 
        # aligns with taking a mean-reverting fade back to our anchored VWAP center line.
        if action == 1:  # Buy Decision
            reward = float(-vwap_dev * 10000.0)
        elif action == 2:  # Sell Decision
            reward = float(vwap_dev * 10000.0)
        else:  # Hold / Flat
            reward = 0.0
            
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Outputs telemetry values directly to standard terminal standard out stream."""
        row = self.data.iloc[self.current_step]
        print(f"Step: {self.current_step} | Price: {row['price']:.5f} | VWAP: {row['vwap']:.5f}")