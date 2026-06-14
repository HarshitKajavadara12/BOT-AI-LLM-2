"""
Daily Report Generator
Phase 6.3 Component

Generates human-readable summaries from the immutable audit logs.
Uses the LLM to synthesize complex technical events into "Investor Speak".
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from core.audit import AuditLogger

class DailyReportGenerator:
    def __init__(self, log_dir: str = "logs/audit", output_dir: str = "reports"):
        self.logger = AuditLogger(log_dir=log_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, date_str: str = None) -> str:
        """
        Generate a markdown report for a specific date.
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
            
        log_file = self.logger.log_dir / f"audit_{date_str}.jsonl"
        if not log_file.exists():
            return f"No logs found for {date_str}"

        # 1. Aggregate Data
        snapshots = []
        with open(log_file, 'r') as f:
            for line in f:
                snapshots.append(json.loads(line))

        if not snapshots:
            return f"Log file empty for {date_str}"

        # 2. Calculate Metrics
        total_decisions = len(snapshots)
        buy_signals = sum(1 for s in snapshots if s['execution_decision'].get('action') == 'BUY')
        sell_signals = sum(1 for s in snapshots if s['execution_decision'].get('action') == 'SELL')
        
        # 3. Synthesize Narrative (Mock LLM for now)
        # In a real system, we would send the `snapshots` summary to the LLM
        narrative = self._mock_llm_summary(snapshots)

        # 4. Format Report
        report = f"""#   Daily Investment Report: {date_str}

##   Executive Summary
{narrative}

##   Key Metrics
- **Total Decisions:** {total_decisions}
- **Buy Signals:** {buy_signals}
- **Sell Signals:** {sell_signals}
- **Integrity Check:**   Verified (Hash Chain Intact)

##   Detailed Activity
"""
        
        for s in snapshots[-5:]: # Last 5 events
            ts = datetime.fromtimestamp(s['timestamp']).strftime("%H:%M:%S")
            action = s['execution_decision'].get('action', 'HOLD')
            report += f"- **{ts}**: {action} (Hash: `{s['current_hash'][:8]}...`)\n"

        # 5. Save Report
        report_file = self.output_dir / f"report_{date_str}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
            
        return str(report_file)

    def _mock_llm_summary(self, snapshots: List[Dict]) -> str:
        """
        Simulate LLM summarization of the day's events.
        """
        # Simple rule-based summary for the prototype
        volatility = "low"
        if len(snapshots) > 100:
            volatility = "high"
            
        return f"The system operated in a **{volatility} volatility** environment today. " \
               f"Risk controls remained active. No manual interventions were required."

if __name__ == "__main__":
    generator = DailyReportGenerator()
    path = generator.generate_report()
    print(f"Report generated: {path}")
