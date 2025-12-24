"""
Tests for the adaptive strategy module.
"""

import json
import pytest
from datetime import datetime, UTC, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.adaptive.asset_profile import (
    AssetProfile,
    AssetProfileManager,
    ProfileConfig,
)
from freqtrade.adaptive.contextual_risk import (
    ContextualRiskFilter,
    ExternalSignal,
    MockSignalProvider,
    RiskFilterConfig,
    RiskLevel,
    RiskModulation,
    SignalType,
)
from freqtrade.adaptive.market_regime import (
    MarketRegime,
    MarketRegimeDetector,
    RegimeConfig,
    RegimeState,
)
from freqtrade.adaptive.trade_explainability import (
    ConfidenceScoreCalculator,
    ExplanationCategory,
    ExplanationEntry,
    ExplainabilityConfig,
    StrategyConfidenceScore,
    TradeExplanation,
    TradeExplainabilityManager,
)


# ============================================================================
# Asset Profile Tests
# ============================================================================

class TestAssetProfile:
    """Tests for AssetProfile dataclass."""
    
    def test_profile_creation(self):
        """Test creating a basic profile."""
        profile = AssetProfile(pair="BTC/USDT")
        assert profile.pair == "BTC/USDT"
        assert profile.volatility == 0.0
        assert profile.win_loss_ratio == 1.0
    
    def test_profile_to_dict(self):
        """Test profile serialization."""
        profile = AssetProfile(
            pair="ETH/USDT",
            volatility=0.05,
            win_loss_ratio=1.5,
        )
        data = profile.to_dict()
        assert data["pair"] == "ETH/USDT"
        assert data["volatility"] == 0.05
        assert "last_updated" in data
    
    def test_profile_from_dict(self):
        """Test profile deserialization."""
        data = {
            "pair": "SOL/USDT",
            "volatility": 0.08,
            "win_loss_ratio": 2.0,
            "last_updated": "2024-01-01T00:00:00+00:00",
        }
        profile = AssetProfile.from_dict(data)
        assert profile.pair == "SOL/USDT"
        assert profile.volatility == 0.08
    
    def test_roi_multiplier(self):
        """Test ROI multiplier calculation."""
        profile = AssetProfile(pair="BTC/USDT", volatility=0.0)
        assert profile.get_roi_multiplier() == 1.0
        
        profile.volatility = 0.1
        multiplier = profile.get_roi_multiplier()
        assert multiplier > 1.0  # Higher volatility = higher ROI targets
    
    def test_stoploss_multiplier(self):
        """Test stoploss multiplier calculation."""
        profile = AssetProfile(pair="BTC/USDT", volatility=0.0)
        assert profile.get_stoploss_multiplier() == 1.0
        
        profile.volatility = 0.15
        multiplier = profile.get_stoploss_multiplier()
        assert multiplier > 1.0  # Higher volatility = wider stops
    
    def test_stake_multiplier(self):
        """Test stake multiplier calculation."""
        # High win rate should increase stake
        profile = AssetProfile(pair="BTC/USDT", win_loss_ratio=2.0)
        assert profile.get_stake_multiplier() > 1.0
        
        # High volatility should reduce stake
        profile = AssetProfile(pair="BTC/USDT", volatility=0.1, win_loss_ratio=1.0)
        assert profile.get_stake_multiplier() < 1.0


class TestAssetProfileManager:
    """Tests for AssetProfileManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def config_enabled(self, temp_dir):
        """Config with profiling enabled."""
        return {
            "adaptive_strategy": {
                "asset_profiles": {
                    "enabled": True,
                    "storage_path": temp_dir,
                    "update_interval_hours": 1,
                }
            }
        }
    
    @pytest.fixture
    def config_disabled(self):
        """Config with profiling disabled."""
        return {"adaptive_strategy": {"asset_profiles": {"enabled": False}}}
    
    def test_manager_disabled(self, config_disabled):
        """Test manager when disabled."""
        manager = AssetProfileManager(config_disabled)
        assert not manager.enabled
        assert manager.get_profile("BTC/USDT") is None
    
    def test_manager_enabled(self, config_enabled, temp_dir):
        """Test manager when enabled."""
        manager = AssetProfileManager(config_enabled)
        assert manager.enabled
        assert Path(temp_dir).exists()
    
    def test_get_or_create_profile(self, config_enabled):
        """Test getting or creating profiles."""
        manager = AssetProfileManager(config_enabled)
        
        # Should create new profile
        profile = manager.get_or_create_profile("BTC/USDT")
        assert profile.pair == "BTC/USDT"
        
        # Should return same profile
        profile2 = manager.get_or_create_profile("BTC/USDT")
        assert profile is profile2
    
    def test_save_and_load_profile(self, config_enabled, temp_dir):
        """Test profile persistence."""
        manager = AssetProfileManager(config_enabled)
        
        # Create and save profile
        profile = AssetProfile(pair="BTC/USDT", volatility=0.05)
        manager._profiles["BTC/USDT"] = profile
        manager.save_profile(profile)
        
        # Verify file exists
        profile_path = Path(temp_dir) / "BTC_USDT.json"
        assert profile_path.exists()
        
        # Load in new manager
        manager2 = AssetProfileManager(config_enabled)
        loaded = manager2.get_profile("BTC/USDT")
        assert loaded is not None
        assert loaded.volatility == 0.05
    
    def test_update_from_dataframe(self, config_enabled):
        """Test updating profile from dataframe."""
        manager = AssetProfileManager(config_enabled)
        
        # Create sample dataframe
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
        df = DataFrame({
            "date": dates,
            "open": np.random.uniform(40000, 45000, 100),
            "high": np.random.uniform(45000, 50000, 100),
            "low": np.random.uniform(35000, 40000, 100),
            "close": np.random.uniform(40000, 45000, 100),
            "volume": np.random.uniform(1000, 5000, 100),
        })
        
        profile = manager.update_profile_from_dataframe("BTC/USDT", df)
        
        assert profile.pair == "BTC/USDT"
        assert profile.volatility > 0
        assert profile.avg_volume > 0
        assert profile.sample_count == 100


# ============================================================================
# Contextual Risk Filter Tests
# ============================================================================

class TestExternalSignal:
    """Tests for ExternalSignal."""
    
    def test_signal_creation(self):
        """Test creating a signal."""
        signal = ExternalSignal(
            signal_type=SignalType.SENTIMENT,
            value=25.0,
            timestamp=datetime.now(UTC),
            source="test",
        )
        assert signal.signal_type == SignalType.SENTIMENT
        assert signal.value == 25.0
    
    def test_signal_expiration(self):
        """Test signal expiration logic."""
        now = datetime.now(UTC)
        
        # Non-expiring signal
        signal = ExternalSignal(
            signal_type=SignalType.SENTIMENT,
            value=50,
            timestamp=now,
            source="test",
            expires_at=None,
        )
        assert not signal.is_expired()
        
        # Expired signal
        signal = ExternalSignal(
            signal_type=SignalType.SENTIMENT,
            value=50,
            timestamp=now - timedelta(hours=2),
            source="test",
            expires_at=now - timedelta(hours=1),
        )
        assert signal.is_expired()
    
    def test_signal_serialization(self):
        """Test signal to/from dict."""
        signal = ExternalSignal(
            signal_type=SignalType.VOLATILITY_INDEX,
            value=0.8,
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            source="test",
            confidence=0.9,
        )
        
        data = signal.to_dict()
        loaded = ExternalSignal.from_dict(data)
        
        assert loaded.signal_type == signal.signal_type
        assert loaded.value == signal.value
        assert loaded.confidence == signal.confidence


class TestRiskModulation:
    """Tests for RiskModulation."""
    
    def test_default_modulation(self):
        """Test default modulation values."""
        mod = RiskModulation()
        assert mod.stake_multiplier == 1.0
        assert mod.block_entries is False
        assert mod.risk_level == RiskLevel.NORMAL
    
    def test_restrictive_modulation(self):
        """Test is_restrictive check."""
        mod = RiskModulation()
        assert not mod.is_restrictive()
        
        mod.block_entries = True
        assert mod.is_restrictive()
        
        mod = RiskModulation(stake_multiplier=0.5)
        assert mod.is_restrictive()


class TestContextualRiskFilter:
    """Tests for ContextualRiskFilter."""
    
    @pytest.fixture
    def config_enabled(self, temp_dir):
        """Config with risk filtering enabled."""
        return {
            "adaptive_strategy": {
                "contextual_risk": {
                    "enabled": True,
                    "providers": [],
                    "signals_dir": temp_dir,
                    "extreme_fear_threshold": 20,
                    "high_fear_threshold": 35,
                }
            }
        }
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_filter_disabled(self):
        """Test filter when disabled."""
        config = {"adaptive_strategy": {"contextual_risk": {"enabled": False}}}
        filter = ContextualRiskFilter(config)
        assert not filter.enabled
        assert filter.get_stake_multiplier() == 1.0
    
    def test_filter_enabled(self, config_enabled):
        """Test filter initialization when enabled."""
        filter = ContextualRiskFilter(config_enabled)
        assert filter.enabled
    
    def test_sentiment_extreme_fear(self, config_enabled):
        """Test extreme fear blocks entries."""
        filter = ContextualRiskFilter(config_enabled)
        
        # Add extreme fear signal
        signal = ExternalSignal(
            signal_type=SignalType.SENTIMENT,
            value=15,  # Below extreme_fear_threshold (20)
            timestamp=datetime.now(UTC),
            source="test",
        )
        filter._current_signals = [signal]
        filter._current_modulation = filter._compute_modulation()
        
        assert filter.should_block_entry()
        assert filter.get_stake_multiplier() < 1.0


class TestMockSignalProvider:
    """Tests for MockSignalProvider."""
    
    def test_load_signals(self, tmp_path):
        """Test loading signals from file."""
        signals_file = tmp_path / "signals.json"
        signals_data = {
            "signals": [
                {
                    "signal_type": "sentiment",
                    "value": 50,
                    "timestamp": "2024-01-01T00:00:00+00:00",
                    "source": "test",
                    "confidence": 0.9,
                }
            ]
        }
        signals_file.write_text(json.dumps(signals_data))
        
        provider = MockSignalProvider(signals_file)
        assert provider.name == "mock"
        
        signals = provider.get_signals(datetime(2024, 1, 1, 12, 0, tzinfo=UTC))
        assert len(signals) == 1
        assert signals[0].value == 50


# ============================================================================
# Market Regime Detection Tests
# ============================================================================

class TestRegimeState:
    """Tests for RegimeState."""
    
    def test_trending_check(self):
        """Test trending regime detection."""
        state = RegimeState(regime=MarketRegime.TRENDING_UP)
        assert state.is_trending()
        
        state = RegimeState(regime=MarketRegime.RANGING)
        assert not state.is_trending()
    
    def test_ranging_check(self):
        """Test ranging regime detection."""
        state = RegimeState(regime=MarketRegime.RANGING)
        assert state.is_ranging()
    
    def test_volatile_check(self):
        """Test volatility regime detection."""
        state = RegimeState(regime=MarketRegime.HIGH_VOLATILITY)
        assert state.is_volatile()


class TestMarketRegimeDetector:
    """Tests for MarketRegimeDetector."""
    
    @pytest.fixture
    def config_enabled(self):
        return {
            "adaptive_strategy": {
                "market_regime": {
                    "enabled": True,
                    "lookback_period": 20,
                }
            }
        }
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample OHLCV dataframe."""
        np.random.seed(42)
        n = 100
        
        # Create trending data
        close = 100 + np.cumsum(np.random.randn(n) * 0.5 + 0.1)
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        open_ = close - np.random.randn(n) * 0.3
        
        return DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.uniform(1000, 5000, n),
        })
    
    def test_detector_disabled(self):
        """Test detector when disabled."""
        config = {"adaptive_strategy": {"market_regime": {"enabled": False}}}
        detector = MarketRegimeDetector(config)
        assert not detector.enabled
        
        state = detector.detect_regime("BTC/USDT", DataFrame())
        assert state.regime == MarketRegime.UNKNOWN
    
    def test_detect_regime(self, config_enabled, sample_dataframe):
        """Test basic regime detection."""
        detector = MarketRegimeDetector(config_enabled)
        
        state = detector.detect_regime("BTC/USDT", sample_dataframe)
        assert state.regime != MarketRegime.UNKNOWN
        assert state.confidence > 0
        assert "adx" in state.metrics
    
    def test_regime_parameters(self, config_enabled, sample_dataframe):
        """Test regime parameter suggestions."""
        detector = MarketRegimeDetector(config_enabled)
        detector.detect_regime("BTC/USDT", sample_dataframe)
        
        params = detector.get_regime_parameters("BTC/USDT")
        assert "roi_multiplier" in params
        assert "stoploss_multiplier" in params


# ============================================================================
# Trade Explainability Tests
# ============================================================================

class TestTradeExplanation:
    """Tests for TradeExplanation."""
    
    def test_explanation_creation(self):
        """Test creating an explanation."""
        exp = TradeExplanation(
            pair="BTC/USDT",
            action="entry",
            direction="long",
        )
        assert exp.pair == "BTC/USDT"
        assert len(exp.entries) == 0
    
    def test_add_indicator(self):
        """Test adding indicator entry."""
        exp = TradeExplanation(pair="BTC/USDT", action="entry", direction="long")
        exp.add_indicator("RSI", 28.5, threshold=30, contribution=0.3)
        
        assert len(exp.entries) == 1
        assert exp.entries[0].category == ExplanationCategory.INDICATOR
        assert exp.entries[0].name == "RSI"
    
    def test_add_regime(self):
        """Test adding regime entry."""
        exp = TradeExplanation(pair="BTC/USDT", action="entry", direction="long")
        exp.add_regime("trending_up", 0.85, contribution=0.2)
        
        assert len(exp.entries) == 1
        assert exp.entries[0].category == ExplanationCategory.MARKET_REGIME
    
    def test_generate_summary(self):
        """Test summary generation."""
        exp = TradeExplanation(pair="BTC/USDT", action="entry", direction="long")
        exp.add_indicator("RSI", 25, contribution=0.4, description="RSI oversold")
        exp.add_indicator("MACD", 0.5, contribution=0.2, description="MACD positive")
        
        summary = exp.generate_summary()
        assert "RSI" in summary
    
    def test_serialization(self):
        """Test to/from dict."""
        exp = TradeExplanation(pair="BTC/USDT", action="entry", direction="long")
        exp.add_indicator("RSI", 30)
        exp.add_regime("trending_up", 0.8)
        
        data = exp.to_dict()
        loaded = TradeExplanation.from_dict(data)
        
        assert loaded.pair == exp.pair
        assert len(loaded.entries) == 2


class TestExplainabilityManager:
    """Tests for TradeExplainabilityManager."""
    
    @pytest.fixture
    def config_enabled(self):
        return {
            "adaptive_strategy": {
                "explainability": {
                    "enabled": True,
                    "log_explanations": False,
                }
            }
        }
    
    def test_manager_disabled(self):
        """Test manager when disabled."""
        config = {"adaptive_strategy": {"explainability": {"enabled": False}}}
        manager = TradeExplainabilityManager(config)
        assert not manager.enabled
    
    def test_start_and_finalize(self, config_enabled):
        """Test starting and finalizing explanations."""
        manager = TradeExplainabilityManager(config_enabled)
        
        # Start explanation
        exp = manager.start_explanation("BTC/USDT", "entry", "long")
        assert exp is not None
        exp.add_indicator("RSI", 25)
        
        # Get pending
        pending = manager.get_pending("BTC/USDT")
        assert pending is exp
        
        # Finalize
        final = manager.finalize_explanation("BTC/USDT", trade_id=123)
        assert final.trade_id == 123
        assert manager.get_pending("BTC/USDT") is None
        
        # Retrieve by trade ID
        stored = manager.get_explanation(123)
        assert stored is final
    
    def test_discard_pending(self, config_enabled):
        """Test discarding pending explanation."""
        manager = TradeExplainabilityManager(config_enabled)
        
        manager.start_explanation("BTC/USDT", "entry", "long")
        assert manager.get_pending("BTC/USDT") is not None
        
        manager.discard_pending("BTC/USDT")
        assert manager.get_pending("BTC/USDT") is None


class TestConfidenceScoreCalculator:
    """Tests for ConfidenceScoreCalculator."""
    
    def test_calculate_score(self):
        """Test confidence score calculation."""
        config = {}
        calculator = ConfidenceScoreCalculator(config)
        
        pair_metrics = {
            "BTC/USDT": {
                "win_rate": 0.6,
                "profit_factor": 1.5,
                "max_drawdown": 0.1,
                "trade_count": 50,
            },
            "ETH/USDT": {
                "win_rate": 0.55,
                "profit_factor": 1.3,
                "max_drawdown": 0.15,
                "trade_count": 45,
            },
        }
        
        score = calculator.calculate_score(pair_metrics)
        
        assert 0 <= score.overall_score <= 1
        assert 0 <= score.stability_score <= 1
        assert "BTC/USDT" in score.pair_scores
    
    def test_warnings_generation(self):
        """Test warning generation for risky results."""
        config = {}
        calculator = ConfidenceScoreCalculator(config)
        
        # Low trade count should generate warning
        pair_metrics = {
            "BTC/USDT": {
                "win_rate": 0.9,
                "profit_factor": 3.0,
                "max_drawdown": 0.05,
                "trade_count": 5,  # Very low
            },
        }
        
        score = calculator.calculate_score(pair_metrics)
        assert any("trade count" in w.lower() for w in score.warnings)
