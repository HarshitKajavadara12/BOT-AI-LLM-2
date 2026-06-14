"""
R Integration Bridge — Documents and provides an optional interface
to the R statistical models in research/.

Error #10: "R integration unused — research/ R scripts"

The R scripts are *research-only* tools for offline statistical modelling:
  - statistical_models.R  — GARCH, rugarch, VAR, cointegration
  - extreme_value.R       — EVT via evd/POT packages
  - copula_analysis.R     — Vine copulas via VineCopula
  - cointegration.R       — Johansen/Engle-Granger tests

These complement (but do not replace) the Python implementations in
core/math_engine/ and core/risk_mathematics/.  They are meant to be run
interactively in RStudio for research validation, not in the live pipeline.

This bridge provides an *optional* subprocess interface should anyone want
to call them from Python (requires R + packages installed).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# R scripts directory relative to project root
_R_DIR = Path(__file__).resolve().parent.parent / "research"

# Available R scripts and their purposes
R_SCRIPTS: Dict[str, str] = {
    "statistical_models.R": "GARCH, VAR, HMM, changepoint detection (rugarch, vars, depmixS4)",
    "extreme_value.R": "GEV / GPD tail estimation (evd, POT, extremevalues)",
    "copula_analysis.R": "Vine copula modelling (VineCopula, copula)",
    "cointegration.R": "Johansen & Engle-Granger cointegration tests (urca, vars)",
}


def r_available() -> bool:
    """Check whether Rscript is available on the system PATH."""
    return shutil.which("Rscript") is not None


def list_r_scripts() -> Dict[str, str]:
    """Return available R scripts and their descriptions."""
    available = {}
    for name, desc in R_SCRIPTS.items():
        path = _R_DIR / name
        if path.exists():
            available[name] = desc
    return available


def run_r_script(script_name: str, *,
                 args: Optional[list] = None,
                 timeout_seconds: int = 120) -> Dict[str, Any]:
    """
    Execute an R script from research/ via subprocess.

    This is OPTIONAL — the live pipeline does not depend on R.
    Useful for offline research validation.

    Args:
        script_name: Filename in research/ (e.g. "extreme_value.R").
        args: Command-line arguments to pass to the script.
        timeout_seconds: Subprocess timeout.

    Returns:
        Dict with keys: success, stdout, stderr, return_code.
    """
    if not r_available():
        return {
            "success": False,
            "stdout": "",
            "stderr": "Rscript not found on PATH. Install R to use research scripts.",
            "return_code": -1,
        }

    script_path = _R_DIR / script_name
    if not script_path.exists():
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Script not found: {script_path}",
            "return_code": -1,
        }

    cmd = ["Rscript", str(script_path)]
    if args:
        cmd.extend(str(a) for a in args)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(_R_DIR),
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"R script timed out after {timeout_seconds}s",
            "return_code": -1,
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "return_code": -1,
        }
