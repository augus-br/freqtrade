"""
Contextual Risk Filter Module

Provides an optional module that consumes aggregated external indicators to modulate
trading risk contextually. This module does NOT directly trigger entries or exits,
only adjusts risk parameters.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from freqtrade.constants import Config


logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk level classifications."""
    VERY_LOW = "very_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


class SignalType(str, Enum):
    """Types of external signals."""
    SENTIMENT = "sentiment"           # Market sentiment (fear/greed)
    SOCIAL_VOLUME = "social_volume"   # Social media activity
    MACRO_EVENT = "macro_event"       # Scheduled macro events
    VOLATILITY_INDEX = "volatility"   # Market-wide volatility
    FUNDING_RATE = "funding_rate"     # Futures funding rates
    CUSTOM = "custom"                 # User-defined signals


@dataclass
class ExternalSignal:
    """
    Represents an external signal used for risk modulation.
    
    Attributes:
        signal_type: Type of the signal
        value: Normalized signal value (typically -1.0 to 1.0 or 0 to 100)
        timestamp: When the signal was generated
        source: Source identifier for the signal
        confidence: Confidence level of the signal (0.0 to 1.0)
        expires_at: When the signal expires and should be ignored
        metadata: Additional signal-specific metadata
    """
    signal_type: SignalType
    value: float
    timestamp: datetime
    source: str
    confidence: float = 1.0
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self, current_time: datetime | None = None) -> bool:
        """Check if the signal has expired."""
        if self.expires_at is None:
            return False
        current_time = current_time or datetime.now(UTC)
        return current_time > self.expires_at
    
    def to_dict(self) -> dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            "signal_type": self.signal_type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "confidence": self.confidence,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExternalSignal":
        """Create signal from dictionary."""
        return cls(
            signal_type=SignalType(data["signal_type"]),
            value=data["value"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data["source"],
            confidence=data.get("confidence", 1.0),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class RiskModulation:
    """
    Risk modulation parameters computed from external signals.
    
    These parameters adjust trading behavior without affecting entry/exit signals.
    
    Attributes:
        stake_multiplier: Multiplier for stake amount (0.0 to 1.0+)
        block_entries: Whether to temporarily block new entries
        stoploss_multiplier: Multiplier for stoploss width (>1.0 = wider stops)
        max_open_trades_multiplier: Multiplier for max open trades
        risk_level: Overall computed risk level
        reasons: List of reasons for the modulation
        timestamp: When the modulation was computed
    """
    stake_multiplier: float = 1.0
    block_entries: bool = False
    stoploss_multiplier: float = 1.0
    max_open_trades_multiplier: float = 1.0
    risk_level: RiskLevel = RiskLevel.NORMAL
    reasons: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def is_restrictive(self) -> bool:
        """Check if the modulation is restricting trading."""
        return (
            self.block_entries or
            self.stake_multiplier < 0.9 or
            self.max_open_trades_multiplier < 0.9
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert modulation to dictionary."""
        return {
            "stake_multiplier": self.stake_multiplier,
            "block_entries": self.block_entries,
            "stoploss_multiplier": self.stoploss_multiplier,
            "max_open_trades_multiplier": self.max_open_trades_multiplier,
            "risk_level": self.risk_level.value,
            "reasons": self.reasons,
            "timestamp": self.timestamp.isoformat(),
        }


class ExternalSignalProvider(ABC):
    """
    Abstract base class for external signal providers.
    
    Implementations can fetch signals from various sources:
    - REST APIs (fear/greed index, social metrics)
    - WebSocket feeds
    - File-based data (for backtesting)
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
    
    @property
    @abstractmethod
    def signal_types(self) -> list[SignalType]:
        """Signal types this provider can supply."""
    
    @abstractmethod
    def get_signals(self, current_time: datetime) -> list[ExternalSignal]:
        """
        Fetch current signals.
        
        Args:
            current_time: Current datetime (for backtest compatibility)
        
        Returns:
            List of current external signals
        """
    
    def is_available(self) -> bool:
        """Check if the provider is available and operational."""
        return True


class MockSignalProvider(ExternalSignalProvider):
    """
    Mock signal provider for backtesting.
    
    Loads signals from a JSON file for deterministic backtest replay.
    """
    
    def __init__(self, signals_file: str | Path) -> None:
        self._signals_file = Path(signals_file)
        self._signals: list[ExternalSignal] = []
        self._load_signals()
    
    @property
    def name(self) -> str:
        return "mock"
    
    @property
    def signal_types(self) -> list[SignalType]:
        return list(SignalType)
    
    def _load_signals(self) -> None:
        """Load signals from file."""
        if not self._signals_file.exists():
            logger.warning(f"Mock signals file not found: {self._signals_file}")
            return
        
        try:
            with open(self._signals_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._signals = [ExternalSignal.from_dict(s) for s in data.get("signals", [])]
            logger.info(f"Loaded {len(self._signals)} mock signals from {self._signals_file}")
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load mock signals: {e}")
    
    def get_signals(self, current_time: datetime) -> list[ExternalSignal]:
        """Get signals valid at the given time."""
        valid_signals = []
        for signal in self._signals:
            if signal.timestamp <= current_time and not signal.is_expired(current_time):
                valid_signals.append(signal)
        return valid_signals


class FearGreedProvider(ExternalSignalProvider):
    """
    Provider for Fear & Greed Index signals.
    
    In live mode, fetches from API. In backtest mode, uses cached data.
    """
    
    def __init__(self, config: Config, cache_path: Path | None = None) -> None:
        self._config = config
        self._cache_path = cache_path or Path("user_data/signals/fear_greed.json")
        self._cached_signals: list[ExternalSignal] = []
        self._last_fetch: datetime | None = None
        self._load_cache()
    
    @property
    def name(self) -> str:
        return "fear_greed"
    
    @property
    def signal_types(self) -> list[SignalType]:
        return [SignalType.SENTIMENT]
    
    def _load_cache(self) -> None:
        """Load cached signals."""
        if self._cache_path.exists():
            try:
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._cached_signals = [ExternalSignal.from_dict(s) for s in data.get("signals", [])]
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load fear/greed cache: {e}")
    
    def get_signals(self, current_time: datetime) -> list[ExternalSignal]:
        """Get fear/greed signals."""
        # For backtesting, use cached historical data
        valid_signals = []
        for signal in self._cached_signals:
            if signal.timestamp <= current_time and not signal.is_expired(current_time):
                valid_signals.append(signal)
        
        # Return the most recent valid signal
        if valid_signals:
            return [max(valid_signals, key=lambda s: s.timestamp)]
        return []


@dataclass
class RiskFilterConfig:
    """Configuration for contextual risk filtering."""
    enabled: bool = False
    providers: list[str] = field(default_factory=list)
    signals_dir: str = "user_data/signals"
    
    # Risk thresholds
    extreme_fear_threshold: int = 20      # Block entries below this
    high_fear_threshold: int = 35         # Reduce stake below this
    high_greed_threshold: int = 75        # Reduce stake above this
    extreme_greed_threshold: int = 85     # Consider reducing exposure
    
    # Modulation settings
    min_stake_multiplier: float = 0.25
    max_stoploss_multiplier: float = 1.5
    
    # Graceful degradation
    fallback_on_error: bool = True
    
    @classmethod
    def from_config(cls, config: Config) -> "RiskFilterConfig":
        """Create RiskFilterConfig from Freqtrade config."""
        adaptive_config = config.get("adaptive_strategy", {})
        risk_config = adaptive_config.get("contextual_risk", {})
        return cls(
            enabled=risk_config.get("enabled", False),
            providers=risk_config.get("providers", []),
            signals_dir=risk_config.get("signals_dir", "user_data/signals"),
            extreme_fear_threshold=risk_config.get("extreme_fear_threshold", 20),
            high_fear_threshold=risk_config.get("high_fear_threshold", 35),
            high_greed_threshold=risk_config.get("high_greed_threshold", 75),
            extreme_greed_threshold=risk_config.get("extreme_greed_threshold", 85),
            min_stake_multiplier=risk_config.get("min_stake_multiplier", 0.25),
            max_stoploss_multiplier=risk_config.get("max_stoploss_multiplier", 1.5),
            fallback_on_error=risk_config.get("fallback_on_error", True),
        )


class ContextualRiskFilter:
    """
    Contextual risk filter that modulates trading parameters based on external signals.
    
    This filter:
    - Does NOT affect entry/exit signal logic
    - Only modulates risk parameters (stake, stoploss, max trades)
    - Supports deterministic backtesting via mock providers
    - Gracefully degrades if signal data is unavailable
    """
    
    def __init__(self, config: Config) -> None:
        self.config = config
        self.risk_config = RiskFilterConfig.from_config(config)
        self._providers: list[ExternalSignalProvider] = []
        self._current_signals: list[ExternalSignal] = []
        self._current_modulation: RiskModulation = RiskModulation()
        self._initialized = False
        
        if self.risk_config.enabled:
            self._init_providers()
            self._initialized = True
    
    @property
    def enabled(self) -> bool:
        """Check if risk filtering is enabled."""
        return self.risk_config.enabled
    
    def _init_providers(self) -> None:
        """Initialize signal providers based on configuration."""
        signals_dir = Path(self.risk_config.signals_dir)
        
        for provider_name in self.risk_config.providers:
            if provider_name == "mock":
                mock_file = signals_dir / "mock_signals.json"
                self._providers.append(MockSignalProvider(mock_file))
            elif provider_name == "fear_greed":
                cache_file = signals_dir / "fear_greed.json"
                self._providers.append(FearGreedProvider(self.config, cache_file))
            else:
                logger.warning(f"Unknown signal provider: {provider_name}")
        
        logger.info(f"Initialized {len(self._providers)} signal providers")
    
    def add_provider(self, provider: ExternalSignalProvider) -> None:
        """Add a custom signal provider."""
        self._providers.append(provider)
        logger.info(f"Added signal provider: {provider.name}")
    
    def update_signals(self, current_time: datetime) -> None:
        """
        Update current signals from all providers.
        
        Args:
            current_time: Current datetime (for backtest compatibility)
        """
        if not self.risk_config.enabled:
            return
        
        self._current_signals = []
        for provider in self._providers:
            try:
                if provider.is_available():
                    signals = provider.get_signals(current_time)
                    self._current_signals.extend(signals)
            except Exception as e:
                logger.warning(f"Failed to get signals from {provider.name}: {e}")
                if not self.risk_config.fallback_on_error:
                    raise
        
        # Compute modulation based on updated signals
        self._current_modulation = self._compute_modulation()
    
    def get_current_modulation(self) -> RiskModulation:
        """Get the current risk modulation parameters."""
        if not self.risk_config.enabled:
            return RiskModulation()
        return self._current_modulation
    
    def get_current_signals(self) -> list[ExternalSignal]:
        """Get all current active signals."""
        return self._current_signals.copy()
    
    def _compute_modulation(self) -> RiskModulation:
        """
        Compute risk modulation based on current signals.
        
        Returns:
            RiskModulation with computed parameters
        """
        modulation = RiskModulation()
        
        if not self._current_signals:
            return modulation
        
        # Process sentiment signals
        sentiment_signals = [s for s in self._current_signals if s.signal_type == SignalType.SENTIMENT]
        if sentiment_signals:
            modulation = self._apply_sentiment_modulation(modulation, sentiment_signals)
        
        # Process volatility signals
        volatility_signals = [s for s in self._current_signals if s.signal_type == SignalType.VOLATILITY_INDEX]
        if volatility_signals:
            modulation = self._apply_volatility_modulation(modulation, volatility_signals)
        
        # Process macro event signals
        macro_signals = [s for s in self._current_signals if s.signal_type == SignalType.MACRO_EVENT]
        if macro_signals:
            modulation = self._apply_macro_modulation(modulation, macro_signals)
        
        modulation.timestamp = datetime.now(UTC)
        return modulation
    
    def _apply_sentiment_modulation(
        self,
        modulation: RiskModulation,
        signals: list[ExternalSignal]
    ) -> RiskModulation:
        """Apply modulation based on sentiment signals (fear/greed)."""
        # Use the most confident/recent signal
        signal = max(signals, key=lambda s: (s.confidence, s.timestamp))
        value = signal.value  # Expected 0-100 scale
        
        if value <= self.risk_config.extreme_fear_threshold:
            # Extreme fear - block entries
            modulation.block_entries = True
            modulation.stake_multiplier = min(modulation.stake_multiplier, self.risk_config.min_stake_multiplier)
            modulation.risk_level = RiskLevel.EXTREME
            modulation.reasons.append(f"Extreme fear ({value:.0f})")
        
        elif value <= self.risk_config.high_fear_threshold:
            # High fear - reduce stake
            fear_factor = (self.risk_config.high_fear_threshold - value) / (
                self.risk_config.high_fear_threshold - self.risk_config.extreme_fear_threshold
            )
            stake_mult = 1.0 - (fear_factor * (1.0 - self.risk_config.min_stake_multiplier))
            modulation.stake_multiplier = min(modulation.stake_multiplier, stake_mult)
            modulation.risk_level = RiskLevel.HIGH
            modulation.reasons.append(f"High fear ({value:.0f})")
        
        elif value >= self.risk_config.extreme_greed_threshold:
            # Extreme greed - reduce exposure
            modulation.stake_multiplier = min(modulation.stake_multiplier, 0.5)
            modulation.max_open_trades_multiplier = min(modulation.max_open_trades_multiplier, 0.75)
            modulation.risk_level = RiskLevel.VERY_HIGH
            modulation.reasons.append(f"Extreme greed ({value:.0f})")
        
        elif value >= self.risk_config.high_greed_threshold:
            # High greed - slight reduction
            greed_factor = (value - self.risk_config.high_greed_threshold) / (
                self.risk_config.extreme_greed_threshold - self.risk_config.high_greed_threshold
            )
            stake_mult = 1.0 - (greed_factor * 0.25)
            modulation.stake_multiplier = min(modulation.stake_multiplier, stake_mult)
            modulation.risk_level = RiskLevel.HIGH
            modulation.reasons.append(f"High greed ({value:.0f})")
        
        return modulation
    
    def _apply_volatility_modulation(
        self,
        modulation: RiskModulation,
        signals: list[ExternalSignal]
    ) -> RiskModulation:
        """Apply modulation based on volatility index signals."""
        signal = max(signals, key=lambda s: (s.confidence, s.timestamp))
        value = signal.value  # Expected normalized value
        
        if value > 0.8:  # Very high volatility
            modulation.stoploss_multiplier = max(
                modulation.stoploss_multiplier,
                min(self.risk_config.max_stoploss_multiplier, 1.0 + value * 0.5)
            )
            modulation.stake_multiplier = min(modulation.stake_multiplier, 0.6)
            modulation.reasons.append(f"High volatility ({value:.2f})")
        elif value > 0.5:  # Elevated volatility
            modulation.stoploss_multiplier = max(modulation.stoploss_multiplier, 1.0 + value * 0.3)
            modulation.reasons.append(f"Elevated volatility ({value:.2f})")
        
        return modulation
    
    def _apply_macro_modulation(
        self,
        modulation: RiskModulation,
        signals: list[ExternalSignal]
    ) -> RiskModulation:
        """Apply modulation based on macro event signals."""
        for signal in signals:
            event_type = signal.metadata.get("event_type", "unknown")
            severity = signal.metadata.get("severity", "medium")
            
            if severity == "high":
                modulation.block_entries = True
                modulation.stake_multiplier = min(modulation.stake_multiplier, 0.25)
                modulation.reasons.append(f"Major macro event: {event_type}")
            elif severity == "medium":
                modulation.stake_multiplier = min(modulation.stake_multiplier, 0.5)
                modulation.reasons.append(f"Macro event: {event_type}")
        
        return modulation
    
    def should_block_entry(self) -> bool:
        """Check if entries should be blocked based on current risk assessment."""
        if not self.risk_config.enabled:
            return False
        return self._current_modulation.block_entries
    
    def get_stake_multiplier(self) -> float:
        """Get the stake multiplier based on current risk assessment."""
        if not self.risk_config.enabled:
            return 1.0
        return self._current_modulation.stake_multiplier
    
    def get_stoploss_multiplier(self) -> float:
        """Get the stoploss multiplier based on current risk assessment."""
        if not self.risk_config.enabled:
            return 1.0
        return self._current_modulation.stoploss_multiplier
