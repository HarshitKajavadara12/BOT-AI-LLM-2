import unittest
from unittest.mock import MagicMock, patch
from execution.order_management.order_management_system import OrderManagementSystem, Order, OrderSide, OrderType
from core.audit import get_audit_logger
import shutil
import tempfile

class TestOMSAudit(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        # Patch the audit logger to use our temp dir
        self.patcher = patch('core.audit.AuditLogger')
        self.MockLogger = self.patcher.start()
        self.mock_logger_instance = self.MockLogger.return_value
        self.mock_logger_instance.log_snapshot.return_value = "test_snapshot_123"
        
        # Patch get_audit_logger to return our mock
        self.get_logger_patcher = patch('execution.order_management.order_management_system.get_audit_logger', return_value=self.mock_logger_instance)
        self.get_logger_patcher.start()
        
        self.oms = OrderManagementSystem("TestOMS")

    def tearDown(self):
        self.patcher.stop()
        self.get_logger_patcher.stop()
        shutil.rmtree(self.test_dir)

    def test_submit_order_logs_snapshot(self):
        """Verify that submitting an order triggers an audit log."""
        order = Order(
            order_id="ord_1",
            client_order_id="cli_1",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        # Submit order
        result = self.oms.submit_order(order)
        
        # Verify success
        self.assertTrue(result)
        
        # Verify audit log was called
        self.mock_logger_instance.log_snapshot.assert_called_once()
        
        # Verify snapshot_id was attached to order
        self.assertEqual(order.snapshot_id, "test_snapshot_123")
        print("[OK] OMS Audit Integration Verified")

if __name__ == '__main__':
    unittest.main()
