"""
Adaptive Strategy Module

This module provides adaptive intelligence mechanisms for Freqtrade strategies,
including per-asset profiling, contextual risk filtering, and market regime detection.
"""

from freqtrade.adaptive.asset_profile import AssetProfile, AssetProfileManager
from freqtrade.adaptive.contextual_risk import (
    ContextualRiskFilter,
    ExternalSignal,
    ExternalSignalProvider,
    RiskModulation,
)
from freqtrade.adaptive.market_regime import MarketRegime, MarketRegimeDetector
from freqtrade.adaptive.mixin import AdaptiveStrategyMixin
from freqtrade.adaptive.trade_explainability import TradeExplanation, TradeExplainabilityManager


__all__ = [
    # Asset Profiling
    "AssetProfile",
    "AssetProfileManager",
    # Contextual Risk
    "ContextualRiskFilter",
    "ExternalSignal",
    "ExternalSignalProvider",
    "RiskModulation",
    # Market Regime
    "MarketRegime",
    "MarketRegimeDetector",
    # Strategy Mixin
    "AdaptiveStrategyMixin",
    # Trade Explainability
    "TradeExplanation",
    "TradeExplainabilityManager",
]
