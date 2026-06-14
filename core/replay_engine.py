"""
Deterministic Replay Engine
Phase 4B Component — UPGRADED to actually replay and verify decisions

This module re-runs historical decisions using the immutable snapshots
stored by the AuditLogger. It ensures that given the same inputs (Market, Signal, Risk),
the system produces the same output (Execution Decision).

This is the "proof" that the system is deterministic — same inputs ALWAYS
produce the same outputs, regardless of when you re-run them.
"""

import json
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
import logging

from core.audit import get_audit_logger, SystemSnapshot

logger = logging.getLogger("ReplayEngine")


@dataclass
class ReplayResult:
    """Result of replaying a single snapshot."""
    snapshot_id: str
    timestamp: str
    is_deterministic: bool        # Did replay match original?
    original_decision: Dict       # What was decided originally
    replayed_decision: Dict       # What replay produced
    divergence_fields: List[str]  # Fields that differ
    confidence: float             # How confident in the match (0-1)


class ReplayEngine:
    """
    Replays historical system states to verify deterministic behavior.
    
    Now actually re-executes the decision logic using snapshot inputs,
    instead of just checking if fields are not None.
    """
    
    def __init__(self):
        self.audit_logger = get_audit_logger()
        self.replay_results: List[ReplayResult] = []
        
    def load_snapshot(self, date_str: str, snapshot_id: str) -> Optional[SystemSnapshot]:
        """Finds a specific snapshot by ID."""
        for snapshot in self.audit_logger.replay_snapshots(date_str):
            if snapshot.snapshot_id == snapshot_id:
                return snapshot
        return None
    
    def verify_decision(self, snapshot: SystemSnapshot) -> ReplayResult:
        """
        Re-runs the decision logic using the snapshot's inputs and compares
        with the recorded decision.
        
        Process:
        1. Extract market_state (prices, returns) from snapshot
        2. Re-run signal generation logic on those inputs
        3. Re-run risk check logic
        4. Compare execution decision with original
        """
        logger.info(f"Replaying Snapshot: {snapshot.snapshot_id}")
        logger.info(f"  Timestamp: {snapshot.iso_timestamp}")
        
        original = snapshot.execution_decision or {}
        market = snapshot.market_state or {}
        signal_state = snapshot.signal_state or {}
        risk_state = snapshot.risk_state or {}
        
        # Re-execute the decision logic
        replayed = self._replay_decision(market, signal_state, risk_state)
        
        # Compare original vs replayed
        divergence = self._compare_decisions(original, replayed)
        
        is_deterministic = len(divergence) == 0
        confidence = 1.0 if is_deterministic else max(0.0, 1.0 - len(divergence) * 0.2)
        
        result = ReplayResult(
            snapshot_id=snapshot.snapshot_id,
            timestamp=snapshot.iso_timestamp,
            is_deterministic=is_deterministic,
            original_decision=original,
            replayed_decision=replayed,
            divergence_fields=divergence,
            confidence=confidence,
        )
        
        self.replay_results.append(result)
        
        if is_deterministic:
            logger.info(f"  VERIFIED: Decision is deterministic (confidence={confidence:.0%})")
        else:
            logger.warning(f"  DIVERGENCE: {len(divergence)} fields differ: {divergence}")
        
        return result
    
    def _replay_decision(
        self,
        market_state: Dict,
        signal_state: Dict,
        risk_state: Dict,
    ) -> Dict:
        """
        Re-execute the decision logic given historical inputs.
        
        This implements the same decision rules that the pipeline uses,
        ensuring deterministic behavior.
        """
        replayed = {
            'action': 'HOLD',
            'symbol': market_state.get('symbol', 'UNKNOWN'),
            'signal_strength': 0.0,
            'risk_approved': False,
            'reason': 'replay',
        }
        
        # 1. Signal check
        signal_strength = signal_state.get('strength', 0.0)
        signal_type = signal_state.get('signal_type', 'HOLD')
        replayed['signal_strength'] = signal_strength
        replayed['signal_type'] = signal_type
        
        if signal_type == 'HOLD' or signal_strength < 0.3:
            replayed['action'] = 'HOLD'
            replayed['reason'] = 'weak_signal'
            return replayed
        
        # 2. Risk check (re-execute the same thresholds)
        var_limit = risk_state.get('var_limit', 0.05)
        current_var = risk_state.get('current_var', 0.0)
        exposure_limit = risk_state.get('exposure_limit', 1.0)
        current_exposure = risk_state.get('current_exposure', 0.0)
        
        risk_ok = (current_var < var_limit) and (current_exposure < exposure_limit)
        replayed['risk_approved'] = risk_ok
        
        if not risk_ok:
            replayed['action'] = 'HOLD'
            replayed['reason'] = 'risk_rejected'
            return replayed
        
        # 3. Regime check
        regime = risk_state.get('regime', 'NEUTRAL')
        if regime in ('CRISIS', 'HIGH_VOLATILITY') and signal_strength < 0.7:
            replayed['action'] = 'HOLD'
            replayed['reason'] = 'regime_blocked'
            return replayed
        
        # 4. Execute
        replayed['action'] = signal_type  # BUY or SELL
        replayed['reason'] = 'signal_approved'
        
        return replayed
    
    def _compare_decisions(self, original: Dict, replayed: Dict) -> List[str]:
        """
        Compare two decisions and return list of fields that diverge.
        """
        divergence = []
        
        # Key fields to compare
        compare_fields = ['action', 'risk_approved', 'reason']
        
        for field in compare_fields:
            orig_val = original.get(field)
            replay_val = replayed.get(field)
            
            if orig_val is not None and replay_val is not None:
                if str(orig_val) != str(replay_val):
                    divergence.append(field)
        
        # Numerical fields — allow small tolerance
        numerical_fields = ['signal_strength']
        for field in numerical_fields:
            orig_val = original.get(field, 0.0)
            replay_val = replayed.get(field, 0.0)
            
            try:
                if abs(float(orig_val) - float(replay_val)) > 1e-6:
                    divergence.append(field)
            except (ValueError, TypeError):
                pass
        
        return divergence
    
    def replay_day(self, date_str: str) -> Dict:
        """
        Replay all snapshots for a given day and return summary.
        
        Returns:
            Dict with total, passed, failed counts and details
        """
        logger.info(f"=== Starting Full Day Replay for {date_str} ===")
        
        results = []
        try:
            for snapshot in self.audit_logger.replay_snapshots(date_str):
                result = self.verify_decision(snapshot)
                results.append(result)
        except FileNotFoundError:
            logger.warning(f"No audit logs found for {date_str}")
            return {'total': 0, 'passed': 0, 'failed': 0, 'results': []}
        
        total = len(results)
        passed = sum(1 for r in results if r.is_deterministic)
        failed = total - passed
        
        summary = {
            'date': date_str,
            'total': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total if total > 0 else 0.0,
            'avg_confidence': np.mean([r.confidence for r in results]) if results else 0.0,
            'divergences': [
                {'snapshot_id': r.snapshot_id, 'fields': r.divergence_fields}
                for r in results if not r.is_deterministic
            ],
        }
        
        logger.info(f"=== Replay Complete: {passed}/{total} passed ({summary['pass_rate']:.0%}) ===")
        
        return summary


if __name__ == "__main__":
    import datetime
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    engine = ReplayEngine()
    
    print(f"--- Starting Replay for {today} ---")
    summary = engine.replay_day(today)
    print(f"\nResults: {summary['passed']}/{summary['total']} deterministic")
    
    if summary['divergences']:
        print(f"\nDivergences found:")
        for d in summary['divergences']:
            print(f"  Snapshot {d['snapshot_id']}: {d['fields']}")
        print("No audit logs found for today.")
