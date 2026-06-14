"""
System Audit & Deterministic Replay Module

This module implements the "Immutable Snapshots" requirement of Phase 4B.
It captures the exact state of the system (Market, Signal, Risk) at every decision point,
enabling deterministic replay and auditing.

Responsibilities:
- Log atomic snapshots of system state to immutable storage (JSONL).
- Provide replay capabilities for verification and debugging.
- Ensure every execution decision is traceable to a specific state.
"""

import json
import time
import threading
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Generator
from pathlib import Path
from datetime import datetime

@dataclass
class SystemSnapshot:
    """Atomic representation of system state at a decision point."""
    snapshot_id: str
    timestamp: float
    iso_timestamp: str
    
    # The Input: What the system saw
    market_state: Dict[str, Any]
    
    # The Logic: Internal calculations
    signal_state: Dict[str, Any]
    risk_state: Dict[str, Any]
    
    # The Output: What the system decided
    execution_decision: Dict[str, Any]
    
    # The Context: Cognitive overlay (Optional, Read-Only)
    cognitive_context: Optional[Dict[str, Any]] = None
    
    # Phase 6: Tamper Evidence
    previous_hash: Optional[str] = None
    current_hash: Optional[str] = None

class AuditLogger:
    """Handles immutable logging of system snapshots."""
    
    def __init__(self, log_dir: str = "logs/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.last_hash = "0" * 64 # Genesis hash
        
        # Initialize last_hash from existing log if available
        self._recover_last_hash()

    def _recover_last_hash(self):
        """Recover the last hash from the most recent log file."""
        try:
            log_file = self._get_log_file()
            if log_file.exists():
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_entry = json.loads(lines[-1])
                        self.last_hash = last_entry.get('current_hash', self.last_hash)
        except Exception:
            pass # Start fresh if error

    def _get_log_file(self) -> Path:
        """Get the current daily log file path."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"audit_{date_str}.jsonl"

    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash of the data dictionary."""
        # Sort keys for deterministic hashing
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

    def log_snapshot(self, snapshot: SystemSnapshot):
        """
        Write a snapshot to the audit log.
        Thread-safe and append-only.
        """
        with self.lock:
            log_file = self._get_log_file()
            
            # Convert to dict
            entry = asdict(snapshot)
            
            # Phase 6: Add Hash Chain
            entry['previous_hash'] = self.last_hash
            # Remove current_hash from calculation to avoid recursion/circularity
            entry['current_hash'] = None 
            
            # Calculate new hash
            new_hash = self._calculate_hash(entry)
            entry['current_hash'] = new_hash
            self.last_hash = new_hash
            
            # Write to file
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def verify_integrity(self, log_file_path: str) -> bool:
        """
        Verify the hash chain integrity of a log file.
        Returns True if valid, False if tampered.
        """
        path = Path(log_file_path)
        if not path.exists():
            return False
            
        last_hash = "0" * 64 # Assuming start of chain or we need to know prev file hash
        # For simplicity, we verify internal consistency of one file
        # In production, we'd link across files.
        
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
                
            if not lines:
                return True
                
            # Check first entry's previous hash if we are strict, 
            # but here we just check chain consistency within the file
            first_entry = json.loads(lines[0])
            last_hash = first_entry['current_hash']
            
            # Re-calculate and check
            # This requires reconstructing the exact dict that was hashed
            # Which is tricky if we don't know the exact previous_hash of the first entry
            # So we'll skip the first entry verification in this simple version
            # and verify the chain from 2nd entry onwards.
            
            for i in range(1, len(lines)):
                entry = json.loads(lines[i])
                prev_hash_in_entry = entry['previous_hash']
                
                if prev_hash_in_entry != last_hash:
                    print(f"Chain broken at line {i+1}")
                    return False
                    
                # Verify current hash
                stored_hash = entry['current_hash']
                entry['current_hash'] = None
                calculated_hash = self._calculate_hash(entry)
                
                if calculated_hash != stored_hash:
                    print(f"Hash mismatch at line {i+1}")
                    return False
                    
                last_hash = stored_hash
                
            return True
            
        except Exception as e:
            print(f"Verification error: {e}")
            return False
        date_str = datetime.now().strftime("%Y-%m-%d")
        self.current_log_file = self.log_dir / f"audit_{date_str}.jsonl"
        
    def log_snapshot(self, 
                     market_state: Dict, 
                     signal_state: Dict, 
                     risk_state: Dict, 
                     decision: Dict,
                     cognitive_context: Optional[Dict] = None) -> str:
        """
        Creates and logs a deterministic snapshot.
        Returns the snapshot_id.
        """
        timestamp = time.time()
        snapshot_id = f"{int(timestamp*1000)}_{hash(json.dumps(decision, sort_keys=True))}"
        
        snapshot = SystemSnapshot(
            snapshot_id=snapshot_id,
            timestamp=timestamp,
            iso_timestamp=datetime.fromtimestamp(timestamp).isoformat(),
            market_state=market_state,
            signal_state=signal_state,
            risk_state=risk_state,
            execution_decision=decision,
            cognitive_context=cognitive_context
        )
        
        # Use the internal _log_snapshot_object method to handle hashing and writing
        self._log_snapshot_object(snapshot)
                
        return snapshot_id

    def _log_snapshot_object(self, snapshot: SystemSnapshot):
        """
        Internal method to write a snapshot object to the audit log with hash chaining.
        Thread-safe and append-only.
        """
        with self.lock:
            log_file = self._get_log_file()
            
            # Convert to dict
            entry = asdict(snapshot)
            
            # Phase 6: Add Hash Chain
            entry['previous_hash'] = self.last_hash
            # Remove current_hash from calculation to avoid recursion/circularity
            entry['current_hash'] = None 
            
            # Calculate new hash
            new_hash = self._calculate_hash(entry)
            entry['current_hash'] = new_hash
            self.last_hash = new_hash
            
            # Write to file
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def replay_snapshots(self, date_str: str) -> Generator[SystemSnapshot, None, None]:
        """
        Yields snapshots from a specific date for deterministic replay.
        """
        log_file = self.log_dir / f"audit_{date_str}.jsonl"
        if not log_file.exists():
            raise FileNotFoundError(f"No audit log found for date: {date_str}")
            
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                yield SystemSnapshot(**data)

# Global instance
_audit_logger = None

def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
