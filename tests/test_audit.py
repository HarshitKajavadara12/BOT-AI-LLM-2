import unittest
import shutil
import tempfile
import json
from pathlib import Path
from core.audit import AuditLogger, SystemSnapshot, get_audit_logger

class TestAuditSystem(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for logs
        self.test_dir = tempfile.mkdtemp()
        self.logger = AuditLogger(log_dir=self.test_dir)
        
    def tearDown(self):
        # Clean up
        shutil.rmtree(self.test_dir)
        
    def test_log_and_replay(self):
        """Verify that snapshots are logged correctly and can be replayed exactly."""
        
        # 1. Create dummy data
        market = {"price": 100, "volume": 500}
        signal = {"alpha": 0.5, "signal": "BUY"}
        risk = {"exposure": 0.1, "limit": 0.2}
        decision = {"action": "BUY", "size": 10}
        
        # 2. Log snapshot
        snapshot_id = self.logger.log_snapshot(market, signal, risk, decision)
        
        # 3. Verify file exists
        log_files = list(Path(self.test_dir).glob("*.jsonl"))
        self.assertEqual(len(log_files), 1)
        
        # 4. Replay and Verify
        import datetime
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        
        snapshots = list(self.logger.replay_snapshots(today))
        self.assertEqual(len(snapshots), 1)
        
        replayed = snapshots[0]
        self.assertEqual(replayed.snapshot_id, snapshot_id)
        self.assertEqual(replayed.market_state, market)
        self.assertEqual(replayed.execution_decision, decision)
        
    def test_immutability_check(self):
        """Ensure the log format is valid JSONL."""
        self.logger.log_snapshot({}, {}, {}, {})
        self.logger.log_snapshot({}, {}, {}, {})
        
        log_files = list(Path(self.test_dir).glob("*.jsonl"))
        with open(log_files[0], 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)
            for line in lines:
                json.loads(line) # Should not raise error

if __name__ == '__main__':
    unittest.main()
