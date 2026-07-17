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
# WINDOWS SSL CORRUPTION MONKEY-PATCH (RUTHLESS EDITION)
# =====================================================================
# This intercepts the root cause of the [ASN1: NOT_ENOUGH_DATA] error 
# across ALL libraries (aiohttp, requests, HuggingFace) by patching 
# the underlying context loader itself.
# =====================================================================
_orig_load_default_certs = ssl.SSLContext.load_default_certs

def _safe_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
    try:
        _orig_load_default_certs(self, purpose)
    except ssl.SSLError:
        print("[SYSTEM] Bypassing corrupted Windows SSL Store for all network requests...")
        try:
            import certifi
            self.load_verify_locations(cafile=certifi.where())
        except ImportError:
            pass

ssl.SSLContext.load_default_certs = _safe_load_default_certs
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
        current_price = obs[0]
        vwap_deviation = obs[1] # Negative if price is BELOW VWAP, Positive if ABOVE VWAP
        
        # =========================================================================
        # THE MASTER's CURRICULUM: VWAP LIQUIDITY SWEEP REWARD FUNCTION
        # =========================================================================
        
        # We define a "significant deviation" as 5 pips (0.00050) away from VWAP
        significant_deviation = 0.00050 

        if action == 1:  # BUY DECISION
            if vwap_deviation < -significant_deviation:
                # PERFECT TRADE: Buying a deep dip below VWAP (Sweeping liquidity)
                reward = 100.0
            elif vwap_deviation > 0:
                # FATAL ERROR: Buying when price is already above VWAP (Chasing)
                reward = -100.0 
            else:
                # Mediocre trade: Buying slightly below VWAP
                reward = 10.0
                
        elif action == 2:  # SELL DECISION
            if vwap_deviation > significant_deviation:
                # PERFECT TRADE: Selling a massive spike above VWAP
                reward = 100.0
            elif vwap_deviation < 0:
                # FATAL ERROR: Selling when price is already below VWAP (Panic selling)
                reward = -100.0
            else:
                # Mediocre trade: Selling slightly above VWAP
                reward = 10.0
                
        else:  # HOLD / FLAT
            # Small penalty to discourage the bot from doing absolutely nothing forever
            reward = -0.1
            
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Outputs telemetry values directly to standard terminal standard out stream."""
        row = self.data.iloc[self.current_step]
        print(f"Step: {self.current_step} | Price: {row['price']:.5f} | VWAP: {row['vwap']:.5f}")