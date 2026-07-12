"""
Platform Architecture: Left Brain (Executive Math & Risk Guardrails)
Environment: Conda fx39 (Python 3.9 win-64)
Dependencies: numpy
"""

import os
from typing import Tuple, Dict, Any


class ExecutiveMathEngine:
    """
    Rigorously enforces position sizing, risk capital limits, 
    and network/broker execution health parameters.
    """

    def __init__(self, initial_balance: float, max_risk_per_trade: float = 0.01, max_daily_drawdown_pct: float = 0.05):
        self.balance = initial_balance
        self.current_equity = initial_balance
        self.max_risk_pct = max_risk_per_trade
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        
        # Absolute operational thresholds
        self.max_allowable_spread = 0.00015  # 1.5 Pips max for intense scalping
        self.max_allowable_ping = 45         # 45ms connection limit
        
        # Session state tracking
        self.daily_starting_balance = initial_balance
        self.total_loss_today = 0.0

    def update_account_metrics(self, current_equity: float) -> None:
        """Dynamically tracks intraday balance fluctuations across tick cycles."""
        self.current_equity = current_equity
        if current_equity < self.daily_starting_balance:
            self.total_loss_today = self.daily_starting_balance - current_equity
        else:
            self.total_loss_today = 0.0

    def assert_risk_guardrails(self, current_spread: float, current_ping: int) -> Tuple[bool, str]:
        """
        Hard boundary system. Returns False with a reason string 
        to immediately terminate order processing if risk thresholds break.
        """
        # 1. Check Circuit Breaker: Daily Drawdown Max Limit
        max_loss_allowed = self.daily_starting_balance * self.max_daily_drawdown_pct
        if self.total_loss_today >= max_loss_allowed:
            return False, "CIRCUIT_BREAKER_TRIGGERED: Daily Drawdown Limit Violated"

        # 2. Check Execution Environment Spread
        if current_spread > self.max_allowable_spread:
            return False, f"EXECUTION_HALTED: Spread too wide ({current_spread:.5f})"

        # 3. Check Network Transport Latency
        if current_ping > self.max_allowable_ping:
            return False, f"EXECUTION_HALTED: High ping latency ({current_ping}ms)"

        return True, "RISK_PROFILE_CLEAR"

    def calculate_precise_lots(self, entry_price: float, stop_loss_price: float, commission_per_lot: float = 3.00) -> float:
        """
        Performs the exact cash allocation risk math.
        Formula: Lots = Risk Capital / ((Stop Loss Distance * Pip Value) + Commission)
        Strictly rounds down to micro-lots (0.01) to eliminate over-exposure.
        """
        pip_distance = abs(entry_price - stop_loss_price)
        
        # Zero distance division defense
        if pip_distance <= 0.0:
            return 0.0

        # Account risk capital parameter calculation
        risk_capital = self.current_equity * self.max_risk_pct
        
        # Standard CME/Interbank currency unit metrics for EUR/USD: 1 Lot = 100,000 base units.
        # A 1-pip movement (0.0001) on a standard lot equals $10.00.
        # Therefore, price distance * 100,000 gives us total cash scale per lot.
        cash_risk_per_lot = (pip_distance * 100000.0) + commission_per_lot
        
        raw_lot_size = risk_capital / cash_risk_per_lot
        
        # Floor calculation to preserve risk integrity down to micro-lot scales
        safe_lots = float(int(raw_lot_size * 100) / 100.0)
        
        # Enforce institutional contract absolute minimum boundaries
        if safe_lots < 0.01:
            return 0.0
            
        return safe_lots