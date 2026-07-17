"""
Platform Architecture: Master Orchestration Conductor Loop
Workspace: Eris
Environment: Conda fx39 (Python 3.9 win-64)
Dependencies: stable-baselines3, pandas, numpy, torch (CPU), python-dotenv
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from dotenv import load_dotenv

# ==========================================================
# SECURE ENVIRONMENT INJECTION
# Loads the hidden .env file so API keys are never hardcoded.
# ==========================================================
load_dotenv()

# Internal Module Imports
from src.rl_env import EurUsdScalpingEnv, fetch_historical_ticks
from src.math_engine import ExecutiveMathEngine
from src.journal import TradingJournal
from src.nlp_senses import FundamentalsEngine
from src.fix_client import initialize_fix_connection


def generate_simulation_data(num_ticks: int = 5000) -> pd.DataFrame:
    """
    Generates synthetic CME Globex MBP-1 Top-of-Book market ticks.
    Guarantees standalone execution verification out-of-the-box.
    """
    print(f"[SIMULATION DATA] Compiling {num_ticks} institutional market ticks...")
    np.random.seed(42)
    
    # Generate a random walk for EUR/USD baseline price
    price_changes = np.random.normal(0, 0.00005, num_ticks)
    prices = 1.08500 + np.cumsum(price_changes)
    
    df = pd.DataFrame({
        'price': prices,
        'size': np.random.randint(1, 50, size=num_ticks),
        'spread': np.random.uniform(0.00005, 0.00012, size=num_ticks), # 0.5 to 1.2 pips
        'sentiment': np.random.uniform(-0.5, 0.5, size=num_ticks)
    })
    
    # Calculate analytical session VWAP tracking
    typical_price = df['price']
    volume = df['size']
    df['vwap'] = (typical_price * volume).cumsum() / volume.cumsum()
    
    return df


def run_platform_core(mode: str = "train") -> None:
    """
    Executes core framework phases: Offline Training or Live Execution.
    Handles component aggregation and cross-engine telemetry passing.
    """
    print("=" * 60)
    print(f"COMMENCING PLATFORM CORE SYSTEM OPERATION MODE: {mode.upper()}")
    print("=" * 60)

    # 1. Initialize Subsystem State Repositories
    journal = TradingJournal()
    math_engine = ExecutiveMathEngine(initial_balance=100000.00)
    
    # Check if a configuration file exists for the FIX engine
    fix_config_path = "config/quickfix.cfg"
    fix_initiator = None
    fix_app = None
    
    if os.path.exists(fix_config_path) and mode == "live":
        print("[SYSTEM] QuickFix configuration detected. Spawning network sockets...")
        fix_initiator, fix_app = initialize_fix_connection(fix_config_path)
        fix_initiator.start()
    else:
        print("[SYSTEM] QuickFix offline or bypassed. Operating in sandbox telemetry.")

    # 2. Acquire High-Resolution Data Arrays
    if mode == "train":
        print("[DATA] Connecting to Databento for historical tick matrices...")
        try:
            db_key = os.getenv("DATABENTO_KEY")
            
            # Fail-safe to ensure the .env file is actually being read
            if not db_key:
                raise ValueError("DATABENTO_KEY is completely missing from the hidden .env file.")

            market_data = fetch_historical_ticks(
                api_key=db_key, 
                start_date="2024-01-02T00:00:00", 
                end_date="2024-01-03T00:00:00",
                symbol="6EH4" # CME Euro FX Futures March 2024 Contract (Highly liquid EUR/USD proxy)
            )
            print(f"[DATA] Successfully fetched {len(market_data)} real market ticks from Databento.")
        except Exception as e:
            print(f"[FATAL DATA ERROR] Failed to fetch from Databento: {e}")
            print("[SYSTEM] Falling back to simulation data to prevent pipeline crash...")
            market_data = generate_simulation_data(num_ticks=10000)
    else:
        print("[DATA] Live Mode: Generating minimal synthetic ticks to keep pipeline hot...")
        market_data = generate_simulation_data(num_ticks=1000)
    
    # 3. Instantiate Custom Gymnasium Environment
    env = EurUsdScalpingEnv(historical_data=market_data)
    
    # Model tracking directories path setup
    model_dir = "models"
    model_path = os.path.join(model_dir, "ppo_eurusd_core")
    os.makedirs(model_dir, exist_ok=True)

    # 4. Phase Branching: Offline Optimization vs Live Stream Integration
    if mode == "train":
        print("[RIGHT BRAIN] Preparing Multi-Layer Perceptron Policy...")
        
        # CPU-optimized hyper-parameters for Stable-Baselines3 PPO
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1,
            tensorboard_log="./logs/"
        )
        
        print("[RIGHT BRAIN] Executing model policy optimization training cycles...")
        model.learn(total_timesteps=500000) # Baseline convergence trial scaling
        model.save(model_path)
        print(f"[RIGHT BRAIN] Training successfully finalized. Model weights committed to: {model_path}")
        
    elif mode == "live":
        # Load local FinBERT context weights for fundamental news injection
        nlp_engine = FundamentalsEngine()
        
        if not os.path.exists(model_path + ".zip"):
            print(f"[FATAL] Pre-trained weights file missing at {model_path}.zip. Run mode='train' first.")
            return
            
        print("[RIGHT BRAIN] Initializing predictive frozen weights context...")
        model = PPO.load(model_path)
        
        print("[SYSTEM] Platform loop online. Transitioning to Live Real-Time Execution...")
        
        # ---------------------------------------------------------
        # TRUE AUTONOMOUS LIVE EXECUTION LOOP
        # ---------------------------------------------------------
        tick_counter = 0
        try:
            while True:
                # 1. FETCH REAL-TIME SENSES
                # -> In a full production deployment, this is where you query your broker's live WebSocket.
                # -> For now, we mock the real-time tick to keep the loop operational and streaming to your UI.
                
                # INJECTING ARTIFICIAL VOLATILITY TO TEST THE BOT'S DISCIPLINE
                tick_counter += 1
                if tick_counter % 30 == 0:
                    current_price = 1.08500 - 0.00060 # Massive 6-pip dip (Bot should BUY)
                elif tick_counter % 50 == 0:
                    current_price = 1.08500 + 0.00060 # Massive 6-pip spike (Bot should SELL)
                else:
                    current_price = 1.08500 + np.random.normal(0, 0.0001) # Normal 1-pip noise (Bot should HOLD)
                
                current_spread = 0.00005 # 0.5 pips
                current_ping_ms = int(np.random.randint(5, 25))
                
                # NLP execution (in production, this reads the latest Reuters/Bloomberg headline)
                live_sentiment = nlp_engine.analyze_headline("No major news.") 
                
                # Approximate VWAP calculation logic for the live stream
                current_vwap = 1.08500 
                vwap_deviation = current_price - current_vwap
                
                # 2. CONSTRUCT THE STATE VECTOR FOR THE BRAIN
                live_obs = np.array([current_price, vwap_deviation, current_spread, live_sentiment], dtype=np.float32)
                
                # 3. PREDICT BEST ACTION (0=Hold, 1=Buy, 2=Sell)
                action, _states = model.predict(live_obs, deterministic=True)
                action = int(action)
                
                # 4. EXECUTION & RISK MANAGEMENT
                if action != 0:
                    action_str = "BUY" if action == 1 else "SELL"
                    
                    is_safe, risk_msg = math_engine.assert_risk_guardrails(
                        current_spread=current_spread, 
                        current_ping=current_ping_ms
                    )
                    
                    if is_safe:
                        simulated_stop_loss = current_price - 0.00150 if action == 1 else current_price + 0.00150
                        calculated_lots = math_engine.calculate_precise_lots(
                            entry_price=current_price,
                            stop_loss_price=simulated_stop_loss
                        )
                        
                        if calculated_lots > 0:
                            journal.log_event(
                                event_type="TRADE_EXECUTION",
                                symbol="EUR/USD",
                                action=action_str,
                                price=current_price,
                                lots=calculated_lots,
                                reason=risk_msg
                            )
                            print(f"[EXECUTE] Signal Verified. Dispatching {action_str} {calculated_lots} Lots at {current_price:.5f}")
                            
                            # Transmit via FIX
                            if fix_app and fix_app.is_logged_on:
                                fix_app.execute_market_order("EUR/USD", action, calculated_lots)
                    else:
                        # Risk desk override
                        journal.log_event(
                            event_type="TRADE_REJECTED",
                            symbol="EUR/USD",
                            action="HOLD",
                            price=current_price,
                            lots=0.0,
                            reason=risk_msg
                        )
                        print(f"[REJECTED] Signal Vetoed by Left Brain. Reason: {risk_msg}")
                
                # Execution pace limiter. Runs 10 times a second.
                time.sleep(0.1) 
                
        except KeyboardInterrupt:
            print("\n[SYSTEM] Manual override detected. Safely powering down execution layers...")
        finally:
            if fix_initiator:
                print("[SYSTEM] Shutting down network initiator sockets safely...")
                fix_initiator.stop()

if __name__ == "__main__":
    # To run the student purely in LIVE mode using the trained brain:
    # (Notice how I commented out the 'train' phase below)
    # run_platform_core(mode="train")
    
    run_platform_core(mode="live")