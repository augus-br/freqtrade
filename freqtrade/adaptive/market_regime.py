"""
Market Regime Detection Module

Classifies market conditions into regimes (trend, range, high volatility)
to allow strategies and asset profiles to react accordingly.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Any

import numpy as np
from pandas import DataFrame, Series

from freqtrade.constants import Config


logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """Market regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    """
    Current market regime state with confidence and metadata.
    
    Attributes:
        regime: The detected market regime
        confidence: Confidence level of the detection (0.0 to 1.0)
        strength: Strength of the regime (e.g., trend strength)
        duration_candles: How long the regime has persisted
        timestamp: When the regime was detected
        metrics: Additional computed metrics
    """
    regime: MarketRegime = MarketRegime.UNKNOWN
    confidence: float = 0.0
    strength: float = 0.0
    duration_candles: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metrics: dict[str, float] = field(default_factory=dict)
    
    def is_trending(self) -> bool:
        """Check if market is in a trending regime."""
        return self.regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN)
    
    def is_ranging(self) -> bool:
        """Check if market is in a ranging regime."""
        return self.regime == MarketRegime.RANGING
    
    def is_volatile(self) -> bool:
        """Check if market is in high volatility regime."""
        return self.regime == MarketRegime.HIGH_VOLATILITY
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "strength": self.strength,
            "duration_candles": self.duration_candles,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
        }


@dataclass
class RegimeConfig:
    """Configuration for market regime detection."""
    enabled: bool = False
    lookback_period: int = 50
    atr_period: int = 14
    adx_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    
    # Thresholds
    adx_trend_threshold: float = 25.0      # ADX above this = trending
    adx_strong_trend: float = 40.0         # Strong trend threshold
    volatility_percentile_high: float = 80  # High volatility percentile
    volatility_percentile_low: float = 20   # Low volatility percentile
    
    @classmethod
    def from_config(cls, config: Config) -> "RegimeConfig":
        """Create RegimeConfig from Freqtrade config."""
        adaptive_config = config.get("adaptive_strategy", {})
        regime_config = adaptive_config.get("market_regime", {})
        return cls(
            enabled=regime_config.get("enabled", False),
            lookback_period=regime_config.get("lookback_period", 50),
            atr_period=regime_config.get("atr_period", 14),
            adx_period=regime_config.get("adx_period", 14),
            bb_period=regime_config.get("bb_period", 20),
            bb_std=regime_config.get("bb_std", 2.0),
            adx_trend_threshold=regime_config.get("adx_trend_threshold", 25.0),
            adx_strong_trend=regime_config.get("adx_strong_trend", 40.0),
            volatility_percentile_high=regime_config.get("volatility_percentile_high", 80),
            volatility_percentile_low=regime_config.get("volatility_percentile_low", 20),
        )


class MarketRegimeDetector:
    """
    Detects and classifies market regimes based on price action and indicators.
    
    Uses multiple indicators for regime classification:
    - ADX for trend detection
    - ATR for volatility
    - Bollinger Bands for ranging detection
    - Price momentum for trend direction
    """
    
    def __init__(self, config: Config) -> None:
        self.config = config
        self.regime_config = RegimeConfig.from_config(config)
        self._pair_states: dict[str, RegimeState] = {}
        self._historical_volatility: dict[str, list[float]] = {}
    
    @property
    def enabled(self) -> bool:
        """Check if regime detection is enabled."""
        return self.regime_config.enabled
    
    def detect_regime(self, pair: str, dataframe: DataFrame) -> RegimeState:
        """
        Detect the current market regime for a trading pair.
        
        Args:
            pair: Trading pair
            dataframe: OHLCV dataframe with sufficient history
        
        Returns:
            RegimeState with detected regime and metrics
        """
        if not self.regime_config.enabled:
            return RegimeState()
        
        if len(dataframe) < self.regime_config.lookback_period:
            return RegimeState(regime=MarketRegime.UNKNOWN, confidence=0.0)
        
        # Compute indicators
        adx = self._calculate_adx(dataframe)
        atr = self._calculate_atr(dataframe)
        bb_width = self._calculate_bb_width(dataframe)
        momentum = self._calculate_momentum(dataframe)
        
        # Store volatility for percentile calculation
        if pair not in self._historical_volatility:
            self._historical_volatility[pair] = []
        self._historical_volatility[pair].append(atr)
        if len(self._historical_volatility[pair]) > 1000:
            self._historical_volatility[pair] = self._historical_volatility[pair][-500:]
        
        # Detect regime
        regime, confidence, strength = self._classify_regime(
            adx, atr, bb_width, momentum, pair
        )
        
        # Track duration
        duration = self._update_duration(pair, regime)
        
        state = RegimeState(
            regime=regime,
            confidence=confidence,
            strength=strength,
            duration_candles=duration,
            timestamp=datetime.now(UTC),
            metrics={
                "adx": adx,
                "atr": atr,
                "bb_width": bb_width,
                "momentum": momentum,
            }
        )
        
        self._pair_states[pair] = state
        return state
    
    def get_regime(self, pair: str) -> RegimeState:
        """Get the last detected regime for a pair."""
        return self._pair_states.get(pair, RegimeState())
    
    def _calculate_adx(self, dataframe: DataFrame) -> float:
        """Calculate Average Directional Index."""
        period = self.regime_config.adx_period
        
        high = dataframe["high"].values
        low = dataframe["low"].values
        close = dataframe["close"].values
        
        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed averages (Wilder's smoothing)
        def wilder_smooth(values: np.ndarray, period: int) -> float:
            if len(values) < period:
                return np.mean(values)
            smoothed = np.mean(values[:period])
            for val in values[period:]:
                smoothed = (smoothed * (period - 1) + val) / period
            return smoothed
        
        atr = wilder_smooth(tr, period)
        plus_di = wilder_smooth(plus_dm, period) / atr * 100 if atr > 0 else 0
        minus_di = wilder_smooth(minus_dm, period) / atr * 100 if atr > 0 else 0
        
        # DX and ADX
        dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
        
        return float(dx)
    
    def _calculate_atr(self, dataframe: DataFrame) -> float:
        """Calculate Average True Range."""
        period = self.regime_config.atr_period
        
        high = dataframe["high"].values[-period-1:]
        low = dataframe["low"].values[-period-1:]
        close = dataframe["close"].values[-period-1:]
        
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        
        # Normalize by price
        avg_price = np.mean(close)
        atr = np.mean(tr) / avg_price if avg_price > 0 else 0
        
        return float(atr)
    
    def _calculate_bb_width(self, dataframe: DataFrame) -> float:
        """Calculate Bollinger Band width (normalized)."""
        period = self.regime_config.bb_period
        std = self.regime_config.bb_std
        
        close = dataframe["close"].values[-period:]
        
        if len(close) < period:
            return 0.0
        
        sma = np.mean(close)
        std_dev = np.std(close)
        
        upper = sma + std * std_dev
        lower = sma - std * std_dev
        
        width = (upper - lower) / sma if sma > 0 else 0
        
        return float(width)
    
    def _calculate_momentum(self, dataframe: DataFrame) -> float:
        """Calculate price momentum (direction indicator)."""
        close = dataframe["close"].values
        
        if len(close) < 20:
            return 0.0
        
        # Calculate returns over different periods
        short_return = (close[-1] / close[-5] - 1) if close[-5] > 0 else 0
        medium_return = (close[-1] / close[-10] - 1) if close[-10] > 0 else 0
        long_return = (close[-1] / close[-20] - 1) if close[-20] > 0 else 0
        
        # Weighted average
        momentum = 0.5 * short_return + 0.3 * medium_return + 0.2 * long_return
        
        return float(momentum)
    
    def _classify_regime(
        self,
        adx: float,
        atr: float,
        bb_width: float,
        momentum: float,
        pair: str
    ) -> tuple[MarketRegime, float, float]:
        """
        Classify market regime based on computed indicators.
        
        Returns:
            Tuple of (regime, confidence, strength)
        """
        # Calculate volatility percentile
        volatility_percentile = self._get_volatility_percentile(pair, atr)
        
        # Determine regime
        if volatility_percentile > self.regime_config.volatility_percentile_high:
            # High volatility regime
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = min(1.0, volatility_percentile / 100)
            strength = atr * 100
        
        elif adx > self.regime_config.adx_trend_threshold:
            # Trending regime
            if momentum > 0:
                regime = MarketRegime.TRENDING_UP
            else:
                regime = MarketRegime.TRENDING_DOWN
            
            # Confidence based on ADX strength
            confidence = min(1.0, adx / self.regime_config.adx_strong_trend)
            strength = adx / 100
        
        elif volatility_percentile < self.regime_config.volatility_percentile_low:
            # Low volatility ranging
            regime = MarketRegime.LOW_VOLATILITY
            confidence = min(1.0, (100 - volatility_percentile) / 100)
            strength = 1.0 - (atr * 100)
        
        else:
            # Ranging regime (moderate volatility, no trend)
            regime = MarketRegime.RANGING
            confidence = min(1.0, (self.regime_config.adx_trend_threshold - adx) / self.regime_config.adx_trend_threshold)
            strength = bb_width
        
        return regime, confidence, strength
    
    def _get_volatility_percentile(self, pair: str, current_atr: float) -> float:
        """Get the percentile rank of current volatility."""
        history = self._historical_volatility.get(pair, [])
        if len(history) < 10:
            return 50.0  # Default to middle
        
        below = sum(1 for v in history if v < current_atr)
        percentile = (below / len(history)) * 100
        return percentile
    
    def _update_duration(self, pair: str, regime: MarketRegime) -> int:
        """Update and return the duration of the current regime."""
        prev_state = self._pair_states.get(pair)
        
        if prev_state is None or prev_state.regime != regime:
            return 1
        
        return prev_state.duration_candles + 1
    
    def get_regime_parameters(self, pair: str) -> dict[str, float]:
        """
        Get suggested parameter adjustments based on current regime.
        
        Returns dictionary with multipliers for various strategy parameters.
        """
        state = self.get_regime(pair)
        
        params = {
            "roi_multiplier": 1.0,
            "stoploss_multiplier": 1.0,
            "trailing_multiplier": 1.0,
            "stake_multiplier": 1.0,
        }
        
        if state.regime == MarketRegime.TRENDING_UP:
            params["roi_multiplier"] = 1.5       # Let profits run
            params["trailing_multiplier"] = 1.2  # Wider trailing
            params["stake_multiplier"] = 1.1     # Slightly larger stakes
        
        elif state.regime == MarketRegime.TRENDING_DOWN:
            params["stake_multiplier"] = 0.8     # Reduce stakes (for spot)
        
        elif state.regime == MarketRegime.RANGING:
            params["roi_multiplier"] = 0.8       # Take profits quicker
            params["stoploss_multiplier"] = 0.9  # Tighter stops
        
        elif state.regime == MarketRegime.HIGH_VOLATILITY:
            params["stoploss_multiplier"] = 1.3  # Wider stops
            params["stake_multiplier"] = 0.7     # Reduce exposure
        
        elif state.regime == MarketRegime.LOW_VOLATILITY:
            params["roi_multiplier"] = 0.7       # Smaller targets
            params["stoploss_multiplier"] = 0.8  # Tighter stops
        
        return params
    
    def populate_regime_column(self, dataframe: DataFrame, pair: str) -> DataFrame:
        """
        Add regime classification columns to a dataframe.
        
        Useful for backtesting analysis and strategy decisions.
        """
        if not self.regime_config.enabled:
            return dataframe
        
        regime_list = []
        confidence_list = []
        
        min_lookback = self.regime_config.lookback_period
        
        for i in range(len(dataframe)):
            if i < min_lookback:
                regime_list.append(MarketRegime.UNKNOWN.value)
                confidence_list.append(0.0)
            else:
                subset = dataframe.iloc[:i+1]
                state = self.detect_regime(pair, subset)
                regime_list.append(state.regime.value)
                confidence_list.append(state.confidence)
        
        dataframe["market_regime"] = regime_list
        dataframe["regime_confidence"] = confidence_list
        
        return dataframe
