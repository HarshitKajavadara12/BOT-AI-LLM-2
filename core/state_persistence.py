"""
QUANTUM-FORGE: State Persistence Engine
=========================================
P1 — System must survive restarts without losing positions or learned weights.

Persists:
  - Open positions and cash balance
  - Trade history
  - ML ensemble model weights (performance-based)
  - Regime detector state
  - Capital allocator weights
  - Portfolio value history
  - Risk gate state

Storage: JSON files in data/state/ (atomic writes with temp file + rename)
"""

import os
import sys
import json
import shutil
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("StatePersistence")

STATE_DIR = Path("./data/state")
STATE_DIR.mkdir(parents=True, exist_ok=True)


class StatePersistence:
    """
    Atomic state save/restore for the QuantumCoreOrchestrator.
    
    Saves state as JSON with atomic file writes (write to temp, then rename).
    Keeps last 5 state snapshots for recovery.
    """
    
    def __init__(self, state_dir: str = None):
        self.state_dir = Path(state_dir) if state_dir else STATE_DIR
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "quantum_core_state.json"
        self.backup_dir = self.state_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def save_state(self, orchestrator) -> bool:
        """
        Save the full orchestrator state to disk.
        
        Args:
            orchestrator: QuantumCoreOrchestrator instance
            
        Returns:
            True if save succeeded
        """
        try:
            state = {
                'version': 2,
                'timestamp': datetime.now().isoformat(),
                'iteration': orchestrator.iteration,
                
                # Financial state
                'initial_capital': orchestrator.initial_capital,
                'cash': orchestrator.cash,
                'positions': orchestrator.positions,
                'trade_history': orchestrator.trade_history[-500:],  # Last 500 trades
                'portfolio_values': list(orchestrator.portfolio_values[-1000:]),
                
                # Performance tracking
                'returns': list(orchestrator._returns),
                'trade_count': orchestrator._trade_count,
                'win_count': orchestrator._win_count,
                
                # Regime
                'current_regime': orchestrator.current_regime.value,
                
                # Risk gate state
                'risk_gate': {
                    'positions': orchestrator.risk_gate.positions,
                    'total_capital': orchestrator.risk_gate.total_capital,
                    'peak_capital': orchestrator.risk_gate.peak_capital,
                    'total_trades': orchestrator.risk_gate.total_trades,
                    'blocked_trades': orchestrator.risk_gate.blocked_trades,
                },
                
                # ML ensemble weights
                'ml_weights': dict(orchestrator.ml_ensemble.model_weights) if orchestrator.ml_ensemble else {},
                'ml_performance': {
                    name: list(perf) 
                    for name, perf in (orchestrator.ml_ensemble.model_performance.items() if orchestrator.ml_ensemble else {})
                },
                
                # Signal generator price history
                'price_history': {
                    symbol: list(prices[-200:])  # Last 200 prices per symbol
                    for symbol, prices in orchestrator.signal_generator.price_history.items()
                },
                
                # Symbols config
                'symbols': orchestrator.symbols,
                'enable_ml': orchestrator.enable_ml,
            }
            
            # Atomic write: temp file → rename
            tmp_file = self.state_file.with_suffix('.tmp')
            with open(tmp_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            # On Windows, remove existing before rename
            if self.state_file.exists():
                self.state_file.unlink()
            tmp_file.rename(self.state_file)
            
            # Rotate backups (keep last 5)
            backup_file = self.backup_dir / f"state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            shutil.copy2(self.state_file, backup_file)
            self._cleanup_backups()
            
            logger.debug(f"State saved: {len(state['positions'])} positions, ${state['cash']:,.2f} cash")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False
    
    def restore_state(self, orchestrator) -> bool:
        """
        Restore orchestrator state from disk.
        
        Args:
            orchestrator: QuantumCoreOrchestrator instance to restore into
            
        Returns:
            True if restore succeeded
        """
        if not self.state_file.exists():
            logger.info("No saved state found — starting fresh")
            return False
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            if state.get('version', 1) < 2:
                logger.warning("Old state format — starting fresh")
                return False
            
            # Restore financial state
            orchestrator.cash = state['cash']
            orchestrator.positions = state['positions']
            orchestrator.trade_history = state.get('trade_history', [])
            orchestrator.portfolio_values = state.get('portfolio_values', [state.get('initial_capital', 100000)])
            orchestrator.iteration = state.get('iteration', 0)
            
            # Restore performance tracking
            orchestrator._returns = deque(state.get('returns', []), maxlen=500)
            orchestrator._trade_count = state.get('trade_count', 0)
            orchestrator._win_count = state.get('win_count', 0)
            
            # Restore regime
            from core.regime_detector import MarketRegime
            regime_val = state.get('current_regime', 'NEUTRAL')
            try:
                orchestrator.current_regime = MarketRegime(regime_val)
            except:
                orchestrator.current_regime = MarketRegime.NEUTRAL
            
            # Restore risk gate
            rg = state.get('risk_gate', {})
            orchestrator.risk_gate.positions = rg.get('positions', {})
            orchestrator.risk_gate.total_capital = rg.get('total_capital', orchestrator.initial_capital)
            orchestrator.risk_gate.peak_capital = rg.get('peak_capital', orchestrator.initial_capital)
            orchestrator.risk_gate.total_trades = rg.get('total_trades', 0)
            orchestrator.risk_gate.blocked_trades = rg.get('blocked_trades', 0)
            
            # Restore ML weights
            if orchestrator.ml_ensemble and state.get('ml_weights'):
                for name, weight in state['ml_weights'].items():
                    if name in orchestrator.ml_ensemble.model_weights:
                        orchestrator.ml_ensemble.model_weights[name] = weight
                
                for name, perf in state.get('ml_performance', {}).items():
                    if name in orchestrator.ml_ensemble.model_performance:
                        orchestrator.ml_ensemble.model_performance[name] = deque(perf, maxlen=100)
            
            # Restore signal generator price history
            for symbol, prices in state.get('price_history', {}).items():
                orchestrator.signal_generator.price_history[symbol] = prices
            
            age = datetime.now() - datetime.fromisoformat(state['timestamp'])
            logger.info(
                f"State restored (saved {age.total_seconds()/60:.0f}m ago): "
                f"{len(orchestrator.positions)} positions, ${orchestrator.cash:,.2f} cash, "
                f"{orchestrator._trade_count} trades, iteration {orchestrator.iteration}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore state: {e}")
            return False
    
    def has_saved_state(self) -> bool:
        """Check if a saved state exists."""
        return self.state_file.exists()
    
    def get_state_info(self) -> Optional[Dict]:
        """Get info about saved state without restoring it."""
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            return {
                'timestamp': state.get('timestamp'),
                'cash': state.get('cash'),
                'n_positions': len(state.get('positions', {})),
                'trade_count': state.get('trade_count', 0),
                'iteration': state.get('iteration', 0),
                'regime': state.get('current_regime'),
            }
        except:
            return None
    
    def _cleanup_backups(self, keep: int = 5):
        """Keep only the most recent N backup files."""
        backups = sorted(self.backup_dir.glob("state_*.json"))
        
        for old_backup in backups[:-keep]:
            try:
                old_backup.unlink()
            except:
                pass
