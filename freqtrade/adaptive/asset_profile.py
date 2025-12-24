"""
Asset Profile Module

Provides per-asset statistical profiling for adaptive strategy parameter modulation.
Each trading pair maintains a lightweight statistical profile used to adjust strategy
parameters at runtime, derived from aggregated historical behavior.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import numpy as np
from pandas import DataFrame

from freqtrade.constants import Config


logger = logging.getLogger(__name__)


@dataclass
class AssetProfile:
    """
    Statistical profile for a trading pair.
    
    Contains aggregated metrics derived from historical data that can be used
    to modulate strategy parameters at runtime.
    
    Attributes:
        pair: Trading pair (e.g., 'BTC/USDT')
        volatility: Rolling volatility measure (standard deviation of returns)
        avg_drawdown: Average drawdown from historical trades
        win_loss_ratio: Ratio of winning to losing trades
        avg_trade_duration_minutes: Average duration of closed trades
        trend_sensitivity: Sensitivity to trending vs ranging markets (0-1)
        avg_volume: Average trading volume
        volume_volatility: Volatility of trading volume
        correlation_btc: Correlation with BTC price movements
        regime_preference: Preferred market regime ('trend', 'range', 'volatile')
        last_updated: Timestamp of last profile update
        sample_count: Number of data points used to compute profile
        custom_metrics: User-defined custom metrics
    """
    pair: str
    volatility: float = 0.0
    avg_drawdown: float = 0.0
    win_loss_ratio: float = 1.0
    avg_trade_duration_minutes: float = 60.0
    trend_sensitivity: float = 0.5
    avg_volume: float = 0.0
    volume_volatility: float = 0.0
    correlation_btc: float = 0.0
    regime_preference: str = "neutral"
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    sample_count: int = 0
    custom_metrics: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert profile to dictionary for serialization."""
        data = asdict(self)
        data["last_updated"] = self.last_updated.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssetProfile":
        """Create profile from dictionary."""
        if "last_updated" in data and isinstance(data["last_updated"], str):
            data["last_updated"] = datetime.fromisoformat(data["last_updated"])
        return cls(**data)
    
    def get_roi_multiplier(self) -> float:
        """
        Calculate ROI multiplier based on asset characteristics.
        
        Higher volatility assets may need wider ROI targets.
        """
        # Base multiplier of 1.0, adjusted by volatility
        if self.volatility > 0:
            # Scale ROI inversely with volatility (high volatility = wider targets)
            return 1.0 + (self.volatility * 0.5)
        return 1.0
    
    def get_stoploss_multiplier(self) -> float:
        """
        Calculate stoploss multiplier based on asset characteristics.
        
        Higher volatility assets may need wider stoplosses to avoid premature exits.
        """
        if self.volatility > 0:
            # Scale stoploss with volatility (high volatility = wider stops)
            return 1.0 + (self.volatility * 0.3)
        return 1.0
    
    def get_stake_multiplier(self) -> float:
        """
        Calculate stake multiplier based on asset performance.
        
        Better performing assets (higher win rate) may deserve larger stakes.
        Lower performing or highly volatile assets should have reduced stakes.
        """
        # Start with win/loss ratio impact
        stake_mult = min(1.5, max(0.5, self.win_loss_ratio))
        
        # Reduce stake for highly volatile assets
        if self.volatility > 0.05:  # 5% volatility threshold
            stake_mult *= (1.0 - (self.volatility - 0.05))
        
        # Reduce stake for high drawdown assets
        if self.avg_drawdown > 0.1:  # 10% drawdown threshold
            stake_mult *= (1.0 - (self.avg_drawdown - 0.1) * 0.5)
        
        return max(0.3, min(1.5, stake_mult))


@dataclass
class ProfileConfig:
    """Configuration for asset profile management."""
    enabled: bool = False
    storage_path: str = "user_data/profiles"
    update_interval_hours: int = 24
    rolling_window_days: int = 30
    min_trades_for_profile: int = 10
    auto_update: bool = True
    
    @classmethod
    def from_config(cls, config: Config) -> "ProfileConfig":
        """Create ProfileConfig from Freqtrade config."""
        adaptive_config = config.get("adaptive_strategy", {})
        profile_config = adaptive_config.get("asset_profiles", {})
        return cls(
            enabled=profile_config.get("enabled", False),
            storage_path=profile_config.get("storage_path", "user_data/profiles"),
            update_interval_hours=profile_config.get("update_interval_hours", 24),
            rolling_window_days=profile_config.get("rolling_window_days", 30),
            min_trades_for_profile=profile_config.get("min_trades_for_profile", 10),
            auto_update=profile_config.get("auto_update", True),
        )


class AssetProfileManager:
    """
    Manages per-asset statistical profiles for adaptive strategy modulation.
    
    Profiles are computed from rolling windows of historical data and cached
    for deterministic behavior during backtesting.
    """
    
    def __init__(self, config: Config) -> None:
        self.config = config
        self.profile_config = ProfileConfig.from_config(config)
        self._profiles: dict[str, AssetProfile] = {}
        self._storage_path = Path(self.profile_config.storage_path)
        self._initialized = False
        
        if self.profile_config.enabled:
            self._ensure_storage_path()
            self._load_profiles()
            self._initialized = True
    
    @property
    def enabled(self) -> bool:
        """Check if asset profiling is enabled."""
        return self.profile_config.enabled
    
    def _ensure_storage_path(self) -> None:
        """Ensure the profile storage directory exists."""
        self._storage_path.mkdir(parents=True, exist_ok=True)
    
    def _get_profile_path(self, pair: str) -> Path:
        """Get the file path for a pair's profile."""
        safe_pair = pair.replace("/", "_")
        return self._storage_path / f"{safe_pair}.json"
    
    def _load_profiles(self) -> None:
        """Load all profiles from storage."""
        if not self._storage_path.exists():
            return
        
        for profile_file in self._storage_path.glob("*.json"):
            try:
                with open(profile_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    profile = AssetProfile.from_dict(data)
                    self._profiles[profile.pair] = profile
                    logger.debug(f"Loaded profile for {profile.pair}")
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Failed to load profile from {profile_file}: {e}")
        
        logger.info(f"Loaded {len(self._profiles)} asset profiles")
    
    def save_profile(self, profile: AssetProfile) -> None:
        """Save a profile to storage."""
        if not self.profile_config.enabled:
            return
        
        profile_path = self._get_profile_path(profile.pair)
        try:
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile.to_dict(), f, indent=2)
            logger.debug(f"Saved profile for {profile.pair}")
        except OSError as e:
            logger.error(f"Failed to save profile for {profile.pair}: {e}")
    
    def get_profile(self, pair: str) -> AssetProfile | None:
        """
        Get the profile for a trading pair.
        
        Returns None if profiling is disabled or no profile exists.
        """
        if not self.profile_config.enabled:
            return None
        return self._profiles.get(pair)
    
    def get_or_create_profile(self, pair: str) -> AssetProfile:
        """
        Get existing profile or create a new empty one.
        """
        if pair in self._profiles:
            return self._profiles[pair]
        
        profile = AssetProfile(pair=pair)
        self._profiles[pair] = profile
        return profile
    
    def update_profile_from_dataframe(
        self,
        pair: str,
        dataframe: DataFrame,
        trades: list[Any] | None = None,
    ) -> AssetProfile:
        """
        Update or create a profile from OHLCV dataframe and trade history.
        
        Args:
            pair: Trading pair
            dataframe: OHLCV dataframe with at least 'close', 'volume' columns
            trades: Optional list of historical trades for the pair
        
        Returns:
            Updated AssetProfile
        """
        profile = self.get_or_create_profile(pair)
        
        if dataframe.empty:
            return profile
        
        # Calculate volatility from returns
        if "close" in dataframe.columns:
            returns = dataframe["close"].pct_change().dropna()
            profile.volatility = float(returns.std())
        
        # Calculate volume metrics
        if "volume" in dataframe.columns:
            profile.avg_volume = float(dataframe["volume"].mean())
            profile.volume_volatility = float(dataframe["volume"].std() / profile.avg_volume) \
                if profile.avg_volume > 0 else 0.0
        
        # Calculate trend sensitivity using ADX-like metric
        if all(col in dataframe.columns for col in ["high", "low", "close"]):
            profile.trend_sensitivity = self._calculate_trend_sensitivity(dataframe)
        
        # Update from trades if available
        if trades:
            profile = self._update_from_trades(profile, trades)
        
        profile.sample_count = len(dataframe)
        profile.last_updated = datetime.now(UTC)
        
        # Save updated profile
        self.save_profile(profile)
        
        return profile
    
    def _calculate_trend_sensitivity(self, dataframe: DataFrame) -> float:
        """
        Calculate trend sensitivity (0-1) based on price action.
        
        Higher values indicate the asset responds better to trending conditions.
        """
        if len(dataframe) < 20:
            return 0.5
        
        # Simple directional movement calculation
        high = dataframe["high"].values
        low = dataframe["low"].values
        close = dataframe["close"].values
        
        # True range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        
        # Directional movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values (simple average for efficiency)
        atr = np.mean(tr) if len(tr) > 0 else 1.0
        plus_di = np.mean(plus_dm) / atr * 100 if atr > 0 else 0
        minus_di = np.mean(minus_dm) / atr * 100 if atr > 0 else 0
        
        # DX (Directional Index)
        dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
        
        # Normalize to 0-1 range
        return float(min(1.0, max(0.0, dx / 100)))
    
    def _update_from_trades(
        self,
        profile: AssetProfile,
        trades: list[Any],
    ) -> AssetProfile:
        """Update profile metrics from trade history."""
        if not trades:
            return profile
        
        wins = 0
        losses = 0
        total_drawdown = 0.0
        total_duration = 0.0
        
        for trade in trades:
            # Handle both Trade objects and dicts
            profit = getattr(trade, "profit_ratio", None) or trade.get("profit_ratio", 0)
            close_date = getattr(trade, "close_date", None) or trade.get("close_date")
            open_date = getattr(trade, "open_date", None) or trade.get("open_date")
            
            if profit is not None:
                if profit > 0:
                    wins += 1
                else:
                    losses += 1
                    total_drawdown += abs(profit)
            
            if close_date and open_date:
                duration = (close_date - open_date).total_seconds() / 60
                total_duration += duration
        
        trade_count = wins + losses
        if trade_count > 0:
            profile.win_loss_ratio = wins / max(1, losses)
            profile.avg_drawdown = total_drawdown / trade_count
            profile.avg_trade_duration_minutes = total_duration / trade_count
        
        return profile
    
    def get_all_profiles(self) -> dict[str, AssetProfile]:
        """Get all loaded profiles."""
        return self._profiles.copy()
    
    def clear_profiles(self) -> None:
        """Clear all profiles from memory (does not delete files)."""
        self._profiles.clear()
    
    def delete_profile(self, pair: str) -> bool:
        """Delete a profile from memory and storage."""
        if pair in self._profiles:
            del self._profiles[pair]
        
        profile_path = self._get_profile_path(pair)
        if profile_path.exists():
            try:
                profile_path.unlink()
                return True
            except OSError as e:
                logger.error(f"Failed to delete profile for {pair}: {e}")
        return False
    
    def should_update_profile(self, pair: str) -> bool:
        """
        Check if a profile should be updated based on update interval.
        """
        profile = self.get_profile(pair)
        if profile is None:
            return True
        
        age_hours = (datetime.now(UTC) - profile.last_updated).total_seconds() / 3600
        return age_hours >= self.profile_config.update_interval_hours
