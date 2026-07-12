"""
Platform Architecture: Nervous System (FIX API Engine)
Environment: Conda fx39 (Python 3.9 win-64)
Dependencies: quickfix (v1.15.1 compiled wheel)
"""

import sys
import os
from typing import Optional
import quickfix as fix

class EurUsdFixClient(fix.Application):
    """
    Core execution connection engine inheriting directly from the 
    QuickFix C++ Application layer. Handles low-latency session 
    management and institutional order routing.
    """

    def __init__(self) -> None:
        super(EurUsdFixClient, self).__init__()
        self.session_id: Optional[fix.SessionID] = None
        self.is_logged_on: bool = False
        
        # Real-time network health metrics tracking
        self.live_spread: float = 0.00005  # Institutional fractions (0.5 pip proxy)
        self.ping_ms: int = 5

    def onCreate(self, session_id: fix.SessionID) -> None:
        """Invoked when the underlying session configuration is initialized."""
        self.session_id = session_id
        print(f"[FIX SYSTEM] Session created: {session_id.toString()}")

    def onLogon(self, session_id: fix.SessionID) -> None:
        """Callback triggered when connection verification succeeds with Prime Broker."""
        self.is_logged_on = True
        print(f"[FIX SYSTEM] Authentication Successful. Session Logon: {session_id.toString()}")

    def onLogout(self, session_id: fix.SessionID) -> None:
        """Callback triggered when connection drops or logout occurs."""
        self.is_logged_on = False
        print(f"[FIX SYSTEM] Session Logout realized: {session_id.toString()}")

    def toAdmin(self, message: fix.Message, session_id: fix.SessionID) -> None:
        """Intercepts administrative traffic (Heartbeats, Logons) before transmission."""
        msg_type = fix.MsgType()
        message.getHeader().getField(msg_type)
        
        # If logging on, ensure password / API key authentication headers match broker spec
        if msg_type.getValue() == fix.MsgType_Logon:
            # Tag 554 = Password, Tag 96 = RawData (Standard Broker Security Specs)
            message.setField(fix.StringField(554, "YOUR_INSTITUTIONAL_PASSWORD"))
            message.setField(fix.StringField(96, "YOUR_SECURE_API_KEY"))

    def fromAdmin(self, message: fix.Message, session_id: fix.SessionID) -> None:
        """Handles inbound administrative messages from the broker matching engine."""
        pass

    def toApp(self, message: fix.Message, session_id: fix.SessionID) -> None:
        """Intercepts application messages (Orders, Cancellations) before transmission."""
        print(f"[FIX OUTBOUND] Outbound traffic dispatched to counterparty.")

    def fromApp(self, message: fix.Message, session_id: fix.SessionID) -> None:
        """
        Receives inbound transactional notifications from execution engines.
        This is where fill notifications are parsed in under a millisecond.
        """
        msg_type = fix.MsgType()
        message.getHeader().getField(msg_type)
        
        # Execution Report (Tag 35 = 8)
        if msg_type.getValue() == fix.MsgType_ExecutionReport:
            exec_type = fix.ExecType()
            message.getField(exec_type)
            
            ord_status = fix.OrdStatus()
            message.getField(ord_status)
            
            # Match status types cleanly without Python 3.10 structural pattern matching
            if ord_status.getValue() == fix.OrdStatus_FILLED:
                avg_px = fix.AvgPx()
                message.getField(avg_px)
                cum_qty = fix.CumQty()
                message.getField(cum_qty)
                print(f"[TRADE EXECUTION] Order Filled! AvgPrice: {avg_px.getValue()}, Lots: {cum_qty.getValue() / 100000}")
                
            elif ord_status.getValue() == fix.OrdStatus_REJECTED:
                print("[TRADE REJECTED] Order flag completely dropped by broker risk desk.")

    def execute_market_order(self, symbol: str, side_flag: int, quantity_lots: float) -> str:
        """
        Translates instructions directly into an immutable FIX message wrapper.
        side_flag inputs: 1 = BUY, 2 = SELL
        """
        if not self.is_logged_on or self.session_id is None:
            return "ABORT_EXECUTION_SESSION_OFFLINE"

        # Initialize a generic transaction message
        order = fix.Message()
        
        # Populate Standard Protocol Header
        header = order.getHeader()
        header.setField(fix.BeginString(fix.BeginString_FIX44)) # Strictly locked to FIX 4.4
        header.setField(fix.MsgType(fix.MsgType_NewOrderSingle))
        
        # Unique Client Order Identifier (Tag 11) Generated per transaction
        cl_ord_id = str(int(pd.Timestamp.now().timestamp() * 1000))
        order.setField(fix.ClOrdID(cl_ord_id))
        
        # Core Transactional Tag Specifications
        order.setField(fix.Symbol(symbol)) # Tag 55
        
        if side_flag == 1:
            order.setField(fix.Side(fix.Side_BUY)) # Tag 54 = 1
        elif side_flag == 2:
            order.setField(fix.Side(fix.Side_SELL)) # Tag 54 = 2
        else:
            return "INVALID_SIDE_SPECIFIED"
            
        # Convert fractional retail lots back to exchange contract contract units (e.g., 1 Lot = 100,000 units)
        contract_units = int(quantity_lots * 100000)
        order.setField(fix.OrderQty(contract_units)) # Tag 38
        
        order.setField(fix.OrdType(fix.OrdType_MARKET)) # Tag 40 = 1
        order.setField(fix.TransactTime()) # Tag 60 = Current Timestamp
        
        # Push message frame down to the open socket buffer channel
        fix.Session.sendToTarget(order, self.session_id)
        return f"ORDER_DISPATCHED_ID_{cl_ord_id}"


def initialize_fix_connection(config_path: str) -> tuple:
    """
    Initializes the C++ execution pipeline wrappers safely.
    Requires a valid quickfix configuration file path.
    """
    try:
        settings = fix.SessionSettings(config_path)
        application = EurUsdFixClient()
        store_factory = fix.FileStoreFactory(settings)
        log_factory = fix.FileLogFactory(settings)
        
        # Initiator wraps everything into a concurrent multi-threaded network socket worker
        initiator = fix.SocketInitiator(application, store_factory, settings, log_factory)
        
        return initiator, application
    except fix.ConfigError as e:
        print(f"[FATAL CONFIG CONFIGURATION CRASH] {str(e)}")
        sys.exit(1)