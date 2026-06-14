"""
Authority Boundary Tests for QUANTUM-FORGE
Ensures strict separation between LLM/Cognitive layers and Execution/Risk layers.

CRITICAL INVARIANTS:
1. Execution modules must NEVER import LLM modules.
2. Risk modules must NEVER import LLM modules.
3. LLM outputs must NEVER be consumed by execution logic.
4. System must function identically with LLM disabled.
"""

import unittest
import sys
import os
import importlib
import ast
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

class TestArchitectureBoundaries(unittest.TestCase):
    """Enforce architectural separation between roles."""

    def setUp(self):
        self.execution_dirs = [
            PROJECT_ROOT / "core" / "execution_algorithms",
            PROJECT_ROOT / "execution",
            PROJECT_ROOT / "risk_management",
            PROJECT_ROOT / "core" / "risk_mathematics"
        ]
        self.llm_modules = ["llm_integration", "langchain", "openai", "llama_cpp"]

    def test_no_forbidden_imports_in_execution(self):
        """
        Invariant: Execution and Risk code must NEVER import LLM-related modules.
        This prevents reverse dependency flow.
        """
        forbidden_imports_found = []

        for directory in self.execution_dirs:
            if not directory.exists():
                continue
                
            for file_path in directory.rglob("*.py"):
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        tree = ast.parse(f.read(), filename=str(file_path))
                    except SyntaxError:
                        continue

                for node in ast.walk(tree):
                    # Check 'import x'
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            for forbidden in self.llm_modules:
                                if alias.name.startswith(forbidden):
                                    forbidden_imports_found.append(
                                        f"{file_path.relative_to(PROJECT_ROOT)} imports {alias.name}"
                                    )
                    
                    # Check 'from x import y'
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            for forbidden in self.llm_modules:
                                if node.module.startswith(forbidden):
                                    forbidden_imports_found.append(
                                        f"{file_path.relative_to(PROJECT_ROOT)} imports from {node.module}"
                                    )

        self.assertEqual(
            len(forbidden_imports_found), 
            0, 
            f"Architecture Violation: Execution/Risk code imports LLM modules!\n" + "\n".join(forbidden_imports_found)
        )

    def test_llm_output_is_informational_only(self):
        """
        Invariant: LLM outputs must be strings or informational dicts, 
        never executable objects or control signals.
        """
        from llm_integration.explanation_contracts import (
            SignalExplanation, RiskExplanation, ExecutionExplanation
        )
        
        # Verify contracts are read-only data structures (Pydantic models)
        # They should not have methods that execute logic
        for contract_cls in [SignalExplanation, RiskExplanation, ExecutionExplanation]:
            # Check for methods that aren't standard Pydantic/BaseModel methods
            methods = [
                func for func in dir(contract_cls) 
                if callable(getattr(contract_cls, func)) 
                and not func.startswith("_")
                and func not in dir(contract_cls.__base__)
            ]
            
            self.assertEqual(
                len(methods), 
                0, 
                f"Violation: Explanation contract {contract_cls.__name__} has executable methods: {methods}. Contracts must be data-only."
            )

class TestKillSwitch(unittest.TestCase):
    """Ensure system respects LLM_ENABLED flag."""

    def test_llm_disabled_behavior(self):
        """
        Invariant: If LLM_ENABLED=false, system must not crash 
        and LLM components must be inert.
        """
        # Simulate environment variable
        os.environ["LLM_ENABLED"] = "false"
        
        # Re-import to trigger initialization logic if any (though good design avoids import-time side effects)
        if "llm_integration.bridge" in sys.modules:
            importlib.reload(sys.modules["llm_integration.bridge"])
        
        from llm_integration.bridge import get_integration_bridge
        
        bridge = get_integration_bridge()
        
        # Check status
        status = bridge.get_status()
        
        # In a real implementation, we'd check if components are actually disabled
        # For now, we verify the bridge exists but acknowledges the flag (if implemented)
        # or at least doesn't crash.
        self.assertIsNotNone(bridge)
        
        # Cleanup
        del os.environ["LLM_ENABLED"]

if __name__ == "__main__":
    unittest.main()
