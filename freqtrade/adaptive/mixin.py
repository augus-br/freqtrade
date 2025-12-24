"""
Adaptive Strategy Mixin

Provides a mixin class that strategies can inherit to gain access to
adaptive features: asset profiling, contextual risk filtering, market regime
detection, and trade explainability.
"""

import logging
from datetime import datetime, UTC
from typing import Any

from pandas import DataFrame

from freqtrade.constants import Config
from freqtrade.adaptive.asset_profile import AssetProfile, AssetProfileManager
from freqtrade.adaptive.contextual_risk import (
    ContextualRiskFilter,
    ExternalSignal,
    RiskModulation,
)
from freqtrade.adaptive.market_regime import MarketRegime, MarketRegimeDetector, RegimeState
from freqtrade.adaptive.trade_explainability import (
    TradeExplanation,
    TradeExplainabilityManager,
)


logger = logging.getLogger(__name__)


class AdaptiveStrategyMixin:
    """
    Mixin class that provides adaptive strategy capabilities.
    
    Strategies can inherit from this mixin to gain:
    - Per-asset parameter profiling
    - Contextual risk filtering
    - Market regime detection
    - Trade explainability
    
    All features are opt-in and configurable via the adaptive_strategy config section.
    
    Example usage:
        class MyStrategy(IStrategy, AdaptiveStrategyMixin):
            def __init__(self, config: Config) -> None:
                super().__init__(config)
                self.init_adaptive(config)
    """
    
    # Adaptive components (initialized by init_adaptive)
    asset_profile_manager: AssetProfileManager | None = None
    risk_filter: ContextualRiskFilter | None = None
    regime_detector: MarketRegimeDetector | None = None
    explainability_manager: TradeExplainabilityManager | None = None
    
    _adaptive_initialized: bool = False
    _adaptive_config: dict[str, Any] = {}
    
    def init_adaptive(self, config: Config) -> None:
        """
        Initialize adaptive strategy components.
        
        Call this in your strategy's __init__ method after super().__init__(config).
        
        Args:
            config: Freqtrade configuration dictionary
        """
        if self._adaptive_initialized:
            return
        
        self._adaptive_config = config.get("adaptive_strategy", {})
        
        if not self._adaptive_config.get("enabled", False):
            logger.info("Adaptive strategy features are disabled")
            self._adaptive_initialized = True
            return
        
        logger.info("Initializing adaptive strategy components")
        
        # Initialize asset profiling
        if self._adaptive_config.get("asset_profiles", {}).get("enabled", False):
            self.asset_profile_manager = AssetProfileManager(config)
            logger.info("Asset profiling enabled")
        
        # Initialize contextual risk filtering
        if self._adaptive_config.get("contextual_risk", {}).get("enabled", False):
            self.risk_filter = ContextualRiskFilter(config)
            logger.info("Contextual risk filtering enabled")
        
        # Initialize market regime detection
        if self._adaptive_config.get("market_regime", {}).get("enabled", False):
            self.regime_detector = MarketRegimeDetector(config)
            logger.info("Market regime detection enabled")
        
        # Initialize trade explainability
        if self._adaptive_config.get("explainability", {}).get("enabled", False):
            self.explainability_manager = TradeExplainabilityManager(config)
            logger.info("Trade explainability enabled")
        
        self._adaptive_initialized = True
        logger.info("Adaptive strategy initialization complete")
    
    @property
    def adaptive_enabled(self) -> bool:
        """Check if any adaptive features are enabled."""
        return self._adaptive_config.get("enabled", False)
    
    # =====================================================================
    # Asset Profile Methods
    # =====================================================================
    
    def get_asset_profile(self, pair: str) -> AssetProfile | None:
        """
        Get the asset profile for a trading pair.
        
        Args:
            pair: Trading pair (e.g., 'BTC/USDT')
        
        Returns:
            AssetProfile if profiling is enabled and profile exists, else None
        """
        if self.asset_profile_manager is None:
            return None
        return self.asset_profile_manager.get_profile(pair)
    
    def update_asset_profile(
        self,
        pair: str,
        dataframe: DataFrame,
        trades: list[Any] | None = None
    ) -> AssetProfile | None:
        """
        Update the asset profile from dataframe and optional trade history.
        
        Args:
            pair: Trading pair
            dataframe: OHLCV dataframe
            trades: Optional list of historical trades
        
        Returns:
            Updated AssetProfile, or None if profiling disabled
        """
        if self.asset_profile_manager is None:
            return None
        return self.asset_profile_manager.update_profile_from_dataframe(pair, dataframe, trades)
    
    def get_profile_adjusted_stoploss(self, pair: str, base_stoploss: float) -> float:
        """
        Get stoploss adjusted by asset profile characteristics.
        
        Args:
            pair: Trading pair
            base_stoploss: Base stoploss value (negative, e.g., -0.1)
        
        Returns:
            Adjusted stoploss value
        """
        profile = self.get_asset_profile(pair)
        if profile is None:
            return base_stoploss
        
        multiplier = profile.get_stoploss_multiplier()
        return base_stoploss * multiplier
    
    def get_profile_adjusted_roi(self, pair: str, base_roi: dict[str, float]) -> dict[str, float]:
        """
        Get ROI table adjusted by asset profile characteristics.
        
        Args:
            pair: Trading pair
            base_roi: Base ROI dictionary
        
        Returns:
            Adjusted ROI dictionary
        """
        profile = self.get_asset_profile(pair)
        if profile is None:
            return base_roi
        
        multiplier = profile.get_roi_multiplier()
        return {k: v * multiplier for k, v in base_roi.items()}
    
    def get_profile_adjusted_stake(self, pair: str, proposed_stake: float) -> float:
        """
        Get stake amount adjusted by asset profile characteristics.
        
        Args:
            pair: Trading pair
            proposed_stake: Originally proposed stake amount
        
        Returns:
            Adjusted stake amount
        """
        profile = self.get_asset_profile(pair)
        if profile is None:
            return proposed_stake
        
        multiplier = profile.get_stake_multiplier()
        return proposed_stake * multiplier
    
    # =====================================================================
    # Contextual Risk Methods
    # =====================================================================
    
    def update_risk_signals(self, current_time: datetime | None = None) -> None:
        """
        Update external risk signals.
        
        Call this in bot_loop_start to ensure signals are current.
        
        Args:
            current_time: Current datetime (for backtest compatibility)
        """
        if self.risk_filter is None:
            return
        
        current_time = current_time or datetime.now(UTC)
        self.risk_filter.update_signals(current_time)
    
    def get_risk_modulation(self) -> RiskModulation:
        """
        Get current risk modulation parameters.
        
        Returns:
            RiskModulation with adjusted parameters
        """
        if self.risk_filter is None:
            return RiskModulation()
        return self.risk_filter.get_current_modulation()
    
    def should_block_entry(self) -> bool:
        """
        Check if entries should be blocked due to risk conditions.
        
        Returns:
            True if entries should be blocked
        """
        if self.risk_filter is None:
            return False
        return self.risk_filter.should_block_entry()
    
    def get_risk_adjusted_stake(self, pair: str, proposed_stake: float) -> float:
        """
        Get stake adjusted by both profile and risk modulation.
        
        Args:
            pair: Trading pair
            proposed_stake: Originally proposed stake
        
        Returns:
            Risk-adjusted stake amount
        """
        # Apply profile adjustment
        stake = self.get_profile_adjusted_stake(pair, proposed_stake)
        
        # Apply risk modulation
        if self.risk_filter is not None:
            risk_mult = self.risk_filter.get_stake_multiplier()
            stake *= risk_mult
        
        return stake
    
    def get_risk_adjusted_stoploss(self, pair: str, base_stoploss: float) -> float:
        """
        Get stoploss adjusted by both profile and risk modulation.
        
        Args:
            pair: Trading pair
            base_stoploss: Base stoploss value
        
        Returns:
            Risk-adjusted stoploss value
        """
        # Apply profile adjustment
        stoploss = self.get_profile_adjusted_stoploss(pair, base_stoploss)
        
        # Apply risk modulation (widen stops during high risk)
        if self.risk_filter is not None:
            risk_mult = self.risk_filter.get_stoploss_multiplier()
            stoploss *= risk_mult
        
        return stoploss
    
    def get_current_signals(self) -> list[ExternalSignal]:
        """
        Get current external signals.
        
        Returns:
            List of current ExternalSignal objects
        """
        if self.risk_filter is None:
            return []
        return self.risk_filter.get_current_signals()
    
    # =====================================================================
    # Market Regime Methods
    # =====================================================================
    
    def detect_market_regime(self, pair: str, dataframe: DataFrame) -> RegimeState:
        """
        Detect the current market regime for a pair.
        
        Args:
            pair: Trading pair
            dataframe: OHLCV dataframe
        
        Returns:
            RegimeState with detected regime and confidence
        """
        if self.regime_detector is None:
            return RegimeState()
        return self.regime_detector.detect_regime(pair, dataframe)
    
    def get_market_regime(self, pair: str) -> RegimeState:
        """
        Get the last detected market regime for a pair.
        
        Args:
            pair: Trading pair
        
        Returns:
            Most recent RegimeState
        """
        if self.regime_detector is None:
            return RegimeState()
        return self.regime_detector.get_regime(pair)
    
    def get_regime_parameters(self, pair: str) -> dict[str, float]:
        """
        Get suggested parameter multipliers based on market regime.
        
        Args:
            pair: Trading pair
        
        Returns:
            Dictionary of parameter multipliers
        """
        if self.regime_detector is None:
            return {}
        return self.regime_detector.get_regime_parameters(pair)
    
    def is_trending(self, pair: str) -> bool:
        """Check if the market is trending for a pair."""
        regime = self.get_market_regime(pair)
        return regime.is_trending()
    
    def is_ranging(self, pair: str) -> bool:
        """Check if the market is ranging for a pair."""
        regime = self.get_market_regime(pair)
        return regime.is_ranging()
    
    def is_volatile(self, pair: str) -> bool:
        """Check if the market is in high volatility for a pair."""
        regime = self.get_market_regime(pair)
        return regime.is_volatile()
    
    # =====================================================================
    # Trade Explainability Methods
    # =====================================================================
    
    def start_trade_explanation(
        self,
        pair: str,
        action: str,
        direction: str = "long"
    ) -> TradeExplanation | None:
        """
        Start building an explanation for a trade.
        
        Args:
            pair: Trading pair
            action: 'entry' or 'exit'
            direction: 'long' or 'short'
        
        Returns:
            TradeExplanation to populate, or None if disabled
        """
        if self.explainability_manager is None:
            return None
        return self.explainability_manager.start_explanation(pair, action, direction)
    
    def get_pending_explanation(self, pair: str) -> TradeExplanation | None:
        """Get pending explanation for a pair."""
        if self.explainability_manager is None:
            return None
        return self.explainability_manager.get_pending(pair)
    
    def finalize_trade_explanation(
        self,
        pair: str,
        trade_id: int | None = None,
        confidence_score: float = 0.0
    ) -> TradeExplanation | None:
        """
        Finalize and store a trade explanation.
        
        Args:
            pair: Trading pair
            trade_id: Associated trade ID
            confidence_score: Confidence in the trade decision
        
        Returns:
            Finalized TradeExplanation
        """
        if self.explainability_manager is None:
            return None
        return self.explainability_manager.finalize_explanation(pair, trade_id, confidence_score)
    
    def discard_explanation(self, pair: str) -> None:
        """Discard pending explanation (trade not executed)."""
        if self.explainability_manager is not None:
            self.explainability_manager.discard_pending(pair)
    
    def get_trade_explanation(self, trade_id: int) -> TradeExplanation | None:
        """Get stored explanation for a trade."""
        if self.explainability_manager is None:
            return None
        return self.explainability_manager.get_explanation(trade_id)
    
    # =====================================================================
    # Combined Adaptive Methods
    # =====================================================================
    
    def get_adaptive_parameters(self, pair: str, dataframe: DataFrame) -> dict[str, Any]:
        """
        Get all adaptive parameter adjustments for a pair.
        
        This is a convenience method that combines profile, risk, and regime adjustments.
        
        Args:
            pair: Trading pair
            dataframe: OHLCV dataframe
        
        Returns:
            Dictionary with all adjustment multipliers and flags
        """
        params: dict[str, Any] = {
            "roi_multiplier": 1.0,
            "stoploss_multiplier": 1.0,
            "stake_multiplier": 1.0,
            "trailing_multiplier": 1.0,
            "block_entry": False,
            "risk_level": "normal",
            "market_regime": "unknown",
            "regime_confidence": 0.0,
        }
        
        # Profile adjustments
        profile = self.get_asset_profile(pair)
        if profile:
            params["roi_multiplier"] *= profile.get_roi_multiplier()
            params["stoploss_multiplier"] *= profile.get_stoploss_multiplier()
            params["stake_multiplier"] *= profile.get_stake_multiplier()
        
        # Risk modulation
        if self.risk_filter is not None:
            modulation = self.get_risk_modulation()
            params["stake_multiplier"] *= modulation.stake_multiplier
            params["stoploss_multiplier"] *= modulation.stoploss_multiplier
            params["block_entry"] = modulation.block_entries
            params["risk_level"] = modulation.risk_level.value
        
        # Regime adjustments
        if self.regime_detector is not None:
            regime_state = self.detect_market_regime(pair, dataframe)
            regime_params = self.get_regime_parameters(pair)
            
            params["roi_multiplier"] *= regime_params.get("roi_multiplier", 1.0)
            params["stoploss_multiplier"] *= regime_params.get("stoploss_multiplier", 1.0)
            params["stake_multiplier"] *= regime_params.get("stake_multiplier", 1.0)
            params["trailing_multiplier"] *= regime_params.get("trailing_multiplier", 1.0)
            params["market_regime"] = regime_state.regime.value
            params["regime_confidence"] = regime_state.confidence
        
        return params
    
    def log_adaptive_state(self, pair: str) -> None:
        """
        Log the current adaptive state for debugging.
        
        Args:
            pair: Trading pair
        """
        if not self.adaptive_enabled:
            return
        
        state_parts = [f"Adaptive state for {pair}:"]
        
        if self.asset_profile_manager is not None:
            profile = self.get_asset_profile(pair)
            if profile:
                state_parts.append(
                    f"  Profile: vol={profile.volatility:.4f}, "
                    f"wr={profile.win_loss_ratio:.2f}, "
                    f"regime_pref={profile.regime_preference}"
                )
        
        if self.risk_filter is not None:
            modulation = self.get_risk_modulation()
            state_parts.append(
                f"  Risk: level={modulation.risk_level.value}, "
                f"stake_mult={modulation.stake_multiplier:.2f}, "
                f"block={modulation.block_entries}"
            )
        
        if self.regime_detector is not None:
            regime = self.get_market_regime(pair)
            state_parts.append(
                f"  Regime: {regime.regime.value} "
                f"(conf={regime.confidence:.2f}, dur={regime.duration_candles})"
            )
        
        logger.info("\n".join(state_parts))
