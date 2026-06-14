"""
QUANTUM-FORGE: Configuration Management
=========================================
Centralised, validated config loading with:
  - YAML as source of truth (system.yaml)
  - Environment variable overrides (QFORGE_*)
  - Runtime hot-reload support
  - Type validation & bounds checking
  - Secret handling (API keys never logged)
"""

import os
import yaml
import logging
from typing import Any, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger("ConfigManager")

DEFAULT_CONFIG_PATH = Path("./system.yaml")


@dataclass
class TradingConfig:
    """Validated trading configuration."""
    mode: str = "paper"                    # paper | live
    symbols: list = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT"
    ])
    initial_capital: float = 10000.0
    max_position_pct: float = 0.05         # max 5% per trade
    min_trade_usd: float = 10.0
    cycle_interval_sec: float = 30.0
    risk_free_rate: float = 0.045


@dataclass
class MLConfig:
    """ML subsystem configuration."""
    feature_dim: int = 32
    ensemble_enabled: bool = True
    gp_enabled: bool = True
    training_lookback_days: int = 90
    svm_n_components: int = 50


@dataclass
class LLMConfig:
    """LLM subsystem configuration."""
    enabled: bool = True
    model_path: str = ""
    max_tokens: int = 512
    temperature: float = 0.3
    authority: str = "read_only"           # read_only | advisory


@dataclass
class InfraConfig:
    """Infrastructure configuration."""
    websocket_enabled: bool = True
    db_backend: str = "duckdb"             # duckdb | timescaledb | sqlite
    redis_url: str = ""
    log_level: str = "INFO"
    health_check_interval: float = 30.0


@dataclass
class SystemConfig:
    """Top-level config aggregating all subsystems."""
    trading: TradingConfig = field(default_factory=TradingConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    infra: InfraConfig = field(default_factory=InfraConfig)
    version: str = "1.0.0"


class ConfigManager:
    """
    Load, validate, and manage system configuration.
    
    Priority (highest first):
      1. Environment variables (QFORGE_TRADING_MODE, QFORGE_LLM_ENABLED, etc.)
      2. system.yaml
      3. Defaults in dataclass definitions
    """

    def __init__(self, config_path: Optional[str] = None):
        self.path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.config = SystemConfig()
        self._raw: Dict = {}
        self._load()

    def _load(self):
        """Load config from YAML + env overrides."""
        # 1. Load YAML
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    self._raw = yaml.safe_load(f) or {}
                logger.info(f"Config loaded from {self.path}")
            except Exception as e:
                logger.warning(f"Failed to load {self.path}: {e}")
                self._raw = {}
        else:
            logger.info(f"No config file at {self.path} — using defaults")

        # 2. Map YAML → dataclasses
        self._map_yaml()

        # 3. Environment overrides
        self._apply_env_overrides()

        # 4. Validate
        self._validate()

        logger.info(f"Config: mode={self.config.trading.mode}, "
                     f"symbols={len(self.config.trading.symbols)}, "
                     f"llm={'ON' if self.config.llm.enabled else 'OFF'}")

    def _map_yaml(self):
        """Map raw YAML dict to typed config objects."""
        trading = self._raw.get("trading", self._raw.get("core", {}))
        ml = self._raw.get("ml", self._raw.get("intelligence", {}))
        llm_sec = self._raw.get("llm", self._raw.get("llm_integration", {}))
        infra = self._raw.get("infrastructure", self._raw.get("infra", {}))

        # Trading
        if isinstance(trading, dict):
            self.config.trading.mode = trading.get("mode", trading.get("execution_mode", "paper"))
            symbols = trading.get("symbols", trading.get("trading_pairs", []))
            if symbols:
                self.config.trading.symbols = symbols
            self.config.trading.initial_capital = float(
                trading.get("initial_capital", trading.get("capital", 10000))
            )
            self.config.trading.max_position_pct = float(
                trading.get("max_position_pct", trading.get("max_allocation", 0.05))
            )
            self.config.trading.cycle_interval_sec = float(
                trading.get("cycle_interval", trading.get("cycle_interval_sec", 30))
            )

        # ML
        if isinstance(ml, dict):
            self.config.ml.feature_dim = int(ml.get("feature_dim", 32))
            self.config.ml.ensemble_enabled = bool(ml.get("ensemble_enabled", True))
            self.config.ml.gp_enabled = bool(ml.get("gp_enabled", True))

        # LLM
        if isinstance(llm_sec, dict):
            self.config.llm.enabled = bool(llm_sec.get("enabled", True))
            self.config.llm.model_path = str(llm_sec.get("model_path", ""))
            self.config.llm.max_tokens = int(llm_sec.get("max_tokens", 512))
            self.config.llm.authority = str(llm_sec.get("authority", "read_only"))

        # Infra
        if isinstance(infra, dict):
            self.config.infra.websocket_enabled = bool(infra.get("websocket_enabled", True))
            self.config.infra.log_level = str(infra.get("log_level", "INFO"))

    def _apply_env_overrides(self):
        """Apply QFORGE_* environment variable overrides."""
        env_map = {
            "QFORGE_MODE": ("trading", "mode"),
            "QFORGE_TRADING_MODE": ("trading", "mode"),
            "QFORGE_CAPITAL": ("trading", "initial_capital", float),
            "QFORGE_LLM_ENABLED": ("llm", "enabled", lambda x: x.lower() in ("1", "true", "yes")),
            "QFORGE_LLM_MODEL": ("llm", "model_path"),
            "QFORGE_LOG_LEVEL": ("infra", "log_level"),
            "QFORGE_WEBSOCKET": ("infra", "websocket_enabled", lambda x: x.lower() in ("1", "true")),
            "QFORGE_FEATURE_DIM": ("ml", "feature_dim", int),
        }

        for env_key, spec in env_map.items():
            val = os.environ.get(env_key)
            if val is None:
                continue

            section_name = spec[0]
            attr_name = spec[1]
            converter = spec[2] if len(spec) > 2 else str

            section = getattr(self.config, section_name)
            try:
                setattr(section, attr_name, converter(val))
                logger.info(f"Config override: {env_key}={val}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid env override {env_key}={val}: {e}")

    def _validate(self):
        """Validate config bounds and consistency."""
        t = self.config.trading
        assert t.mode in ("paper", "live"), f"Invalid mode: {t.mode}"
        assert 0 < t.max_position_pct <= 0.5, f"max_position_pct out of range: {t.max_position_pct}"
        assert t.initial_capital > 0, f"Capital must be positive: {t.initial_capital}"
        assert t.cycle_interval_sec >= 1, f"Cycle too fast: {t.cycle_interval_sec}"

        m = self.config.ml
        assert m.feature_dim in (20, 32, 64), f"Unusual feature_dim: {m.feature_dim}"

        l = self.config.llm
        assert l.authority in ("read_only", "advisory"), f"LLM authority: {l.authority}"

    def get(self, key: str, default: Any = None) -> Any:
        """Dot-notation access: config.get('trading.mode')"""
        parts = key.split(".")
        obj = self.config
        for p in parts:
            obj = getattr(obj, p, None)
            if obj is None:
                return default
        return obj

    def reload(self):
        """Hot-reload config from disk."""
        logger.info("Reloading configuration...")
        self._load()

    def to_dict(self) -> Dict:
        """Export config as dict (secrets redacted)."""
        from dataclasses import asdict
        d = asdict(self.config)
        # Redact secrets
        if "llm" in d and "model_path" in d["llm"]:
            path = d["llm"]["model_path"]
            if path:
                d["llm"]["model_path"] = f"...{path[-20:]}" if len(path) > 20 else path
        return d


# Singleton
_config_instance: Optional[ConfigManager] = None


def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """Get or create the global config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager(config_path)
    return _config_instance
