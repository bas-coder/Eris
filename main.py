"""
Platform Architecture: Master Orchestration Conductor Loop
Workspace: Eris
Environment: Conda fx39 (Python 3.9 win-64)
Dependencies: stable-baselines3, pandas, numpy, torch (CPU)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

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
            # ---> PASTE YOUR DATABENTO API KEY IN THE QUOTES BELOW <---
            # We start with 1 week of data to ensure the download pipeline is flawless before pulling months of data.
            market_data = fetch_historical_ticks(
                api_key=os.getenv('DATABENTO_KEY'), 
                start_date="2024-01-01", 
                end_date="2024-01-07",
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
        model.learn(total_timesteps=20000) # Baseline convergence trial scaling
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
        
        obs, info = env.reset()
        done = False
        step_idx = 0
        
        print("[SYSTEM] Platform loop online. Awaiting data stream interrupts...")
        
        # Real-time ticking mock evaluation loop
        while not done and step_idx < 100:
            # Predict best directional action step using frozen RL weights
            action, _states = model.predict(obs, deterministic=True)
            action = int(action)
            
            # Fetch environmental metrics from current step vector
            current_price = float(obs[0])
            current_spread = float(obs[2])
            
            # Simulated ping variable proxy for retail vs institutional checks
            mock_ping_ms = int(np.random.randint(10, 25)) 
            
            if action != 0: # 1 = BUY, 2 = SELL
                # Left Brain validation pass before transaction commitment
                is_safe, risk_msg = math_engine.assert_risk_guardrails(
                    current_spread=current_spread, 
                    current_ping=mock_ping_ms
                )
                
                if is_safe:
                    # Dynamically compute target position size using account mathematical equity
                    simulated_stop_loss = current_price - 0.00150 if action == 1 else current_price + 0.00150
                    calculated_lots = math_engine.calculate_precise_lots(
                        entry_price=current_price,
                        stop_loss_price=simulated_stop_loss
                    )
                    
                    action_str = "BUY" if action == 1 else "SELL"
                    
                    if calculated_lots > 0:
                        # Log clearance to structural system telemetry ledger
                        journal.log_event(
                            event_type="TRADE_EXECUTION",
                            symbol="EUR/USD",
                            action=action_str,
                            price=current_price,
                            lots=calculated_lots,
                            reason=risk_msg
                        )
                        print(f"[EXECUTE] Order Dispatched: {action_str} {calculated_lots} Lots at {current_price:.5f}")
                        
                        # Direct interaction with the open network socket if connected live
                        if fix_app and fix_app.is_logged_on:
                            fix_app.execute_market_order("EUR/USD", action, calculated_lots)
                else:
                    # The risk manager successfully blocks execution due to adverse metrics
                    journal.log_event(
                        event_type="TRADE_REJECTED",
                        symbol="EUR/USD",
                        action="HOLD",
                        price=current_price,
                        lots=0.0,
                        reason=risk_msg
                    )
                    print(f"[REJECTED] Left Brain Overrode Signal. Reason: {risk_msg}")
            
            # Progress environment tick state forward
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_idx += 1
            time.sleep(0.01) # Execution pace limiter to protect thread cycle consumption
            
        print("[SYSTEM] Operational evaluation routine completed safely.")
        
    if fix_initiator:
        print("[SYSTEM] Shutting down network initiator sockets safely...")
        fix_initiator.stop()


if __name__ == "__main__":
    # Standard operational run pipeline
    # Phase A: Train the core brain parameters
    run_platform_core(mode="train")
    
    # Phase B: Route the trained model down to structural execution validation layers
    run_platform_core(mode="live")