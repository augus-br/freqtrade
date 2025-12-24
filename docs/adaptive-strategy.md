# Adaptive Strategy Features

Freqtrade supports adaptive intelligence mechanisms that allow strategies to dynamically adjust parameters per trading pair and incorporate external contextual risk signals.

These features are **fully opt-in** and preserve:
- Deterministic behavior
- Backtesting reproducibility  
- Full compatibility with hyperopt

## Overview

The adaptive strategy system consists of four main components:

1. **Per-Asset Profiling** - Maintain pair-specific statistical profiles
2. **Contextual Risk Filtering** - Modulate risk based on external signals
3. **Market Regime Detection** - Classify market conditions
4. **Trade Explainability** - Attach structured reasons to trades

## Quick Start

### 1. Enable in Configuration

Add the `adaptive_strategy` section to your config:

```json
{
    "adaptive_strategy": {
        "enabled": true,
        "asset_profiles": {
            "enabled": true
        },
        "contextual_risk": {
            "enabled": true,
            "providers": ["mock"]
        },
        "market_regime": {
            "enabled": true
        },
        "explainability": {
            "enabled": true
        }
    }
}
```

### 2. Inherit from AdaptiveStrategyMixin

```python
from freqtrade.strategy import IStrategy
from freqtrade.adaptive import AdaptiveStrategyMixin

class MyStrategy(IStrategy, AdaptiveStrategyMixin):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.init_adaptive(config)  # Initialize adaptive features
```

### 3. Use Adaptive Methods

```python
def populate_indicators(self, dataframe, metadata):
    pair = metadata['pair']
    
    # Update asset profile
    self.update_asset_profile(pair, dataframe)
    
    # Detect market regime
    regime = self.detect_market_regime(pair, dataframe)
    
    return dataframe

def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
    # Get stoploss adjusted by profile and risk conditions
    return self.get_risk_adjusted_stoploss(pair, self.stoploss)
```

---

## Per-Asset Profiling

Asset profiling maintains lightweight statistical profiles for each trading pair, derived from historical data.

### Configuration

```json
{
    "adaptive_strategy": {
        "asset_profiles": {
            "enabled": true,
            "storage_path": "user_data/profiles",
            "update_interval_hours": 24,
            "rolling_window_days": 30,
            "min_trades_for_profile": 10,
            "auto_update": true
        }
    }
}
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `enabled` | Enable asset profiling | `false` |
| `storage_path` | Directory to store profile JSON files | `user_data/profiles` |
| `update_interval_hours` | Hours between profile updates | `24` |
| `rolling_window_days` | Days of history for profile calculation | `30` |
| `min_trades_for_profile` | Minimum trades before profile is used | `10` |
| `auto_update` | Auto-update profiles during trading | `true` |

### Profile Metrics

Each profile contains:

| Metric | Description |
|--------|-------------|
| `volatility` | Standard deviation of returns |
| `avg_drawdown` | Average drawdown from trades |
| `win_loss_ratio` | Ratio of winning to losing trades |
| `avg_trade_duration_minutes` | Average trade duration |
| `trend_sensitivity` | Response to trending conditions (0-1) |
| `avg_volume` | Average trading volume |
| `volume_volatility` | Volatility of trading volume |

### Usage in Strategy

```python
# Get profile for a pair
profile = self.get_asset_profile(pair)
if profile:
    print(f"Volatility: {profile.volatility}")
    print(f"Win/Loss Ratio: {profile.win_loss_ratio}")

# Get profile-adjusted parameters
adjusted_stoploss = self.get_profile_adjusted_stoploss(pair, base_stoploss)
adjusted_roi = self.get_profile_adjusted_roi(pair, base_roi)
adjusted_stake = self.get_profile_adjusted_stake(pair, proposed_stake)
```

---

## Contextual Risk Filtering

Risk filtering modulates trading parameters based on external signals (sentiment, volatility, macro events).

!!! important
    Risk filtering does NOT alter entry/exit signals. It only adjusts:
    - Stake size
    - Stoploss width
    - Entry blocking (temporarily)
    - Maximum open trades

### Configuration

```json
{
    "adaptive_strategy": {
        "contextual_risk": {
            "enabled": true,
            "providers": ["mock", "fear_greed"],
            "signals_dir": "user_data/signals",
            "extreme_fear_threshold": 20,
            "high_fear_threshold": 35,
            "high_greed_threshold": 75,
            "extreme_greed_threshold": 85,
            "min_stake_multiplier": 0.25,
            "max_stoploss_multiplier": 1.5,
            "fallback_on_error": true
        }
    }
}
```

### Signal Providers

| Provider | Description |
|----------|-------------|
| `mock` | Loads signals from `signals_dir/mock_signals.json` for backtesting |
| `fear_greed` | Fear & Greed Index (cached data) |

### Creating Mock Signals for Backtesting

Create `user_data/signals/mock_signals.json`:

```json
{
    "signals": [
        {
            "signal_type": "sentiment",
            "value": 25,
            "timestamp": "2024-01-15T00:00:00+00:00",
            "source": "mock_fear_greed",
            "confidence": 0.85,
            "expires_at": "2024-01-16T00:00:00+00:00",
            "metadata": {"classification": "extreme_fear"}
        },
        {
            "signal_type": "macro_event",
            "value": 1.0,
            "timestamp": "2024-03-01T12:00:00+00:00",
            "source": "mock_calendar",
            "confidence": 1.0,
            "expires_at": "2024-03-01T18:00:00+00:00",
            "metadata": {
                "event_type": "FOMC",
                "severity": "high"
            }
        }
    ]
}
```

### Signal Types

| Type | Description | Value Range |
|------|-------------|-------------|
| `sentiment` | Fear/Greed index | 0-100 |
| `volatility` | Volatility index | 0.0-1.0 |
| `social_volume` | Social media activity | Multiplier |
| `macro_event` | Scheduled events | 0.0-1.0 |

### Usage in Strategy

```python
def bot_loop_start(self, current_time, **kwargs):
    # Update signals at the start of each loop
    self.update_risk_signals(current_time)

def confirm_trade_entry(self, pair, order_type, amount, rate, **kwargs):
    # Check if entries are blocked
    if self.should_block_entry():
        return False
    return True

def custom_stake_amount(self, pair, proposed_stake, **kwargs):
    # Get risk-adjusted stake
    return self.get_risk_adjusted_stake(pair, proposed_stake)
```

---

## Market Regime Detection

Classifies market conditions into regimes (trending, ranging, high volatility) using technical indicators.

### Configuration

```json
{
    "adaptive_strategy": {
        "market_regime": {
            "enabled": true,
            "lookback_period": 50,
            "atr_period": 14,
            "adx_period": 14,
            "adx_trend_threshold": 25.0,
            "adx_strong_trend": 40.0,
            "volatility_percentile_high": 80,
            "volatility_percentile_low": 20
        }
    }
}
```

### Regime Types

| Regime | Description |
|--------|-------------|
| `trending_up` | Upward trend (ADX > threshold, positive momentum) |
| `trending_down` | Downward trend (ADX > threshold, negative momentum) |
| `ranging` | No clear trend (low ADX) |
| `high_volatility` | High volatility period |
| `low_volatility` | Low volatility period |
| `breakout` | Potential breakout forming |

### Usage in Strategy

```python
def populate_entry_trend(self, dataframe, metadata):
    pair = metadata['pair']
    
    # Detect regime
    regime = self.detect_market_regime(pair, dataframe)
    
    # Adjust entry logic based on regime
    if self.is_trending(pair):
        # Use trend-following entries
        rsi_threshold = 40
    elif self.is_ranging(pair):
        # Use mean-reversion entries
        rsi_threshold = 25
    else:
        rsi_threshold = 30
    
    # Get regime-based parameter multipliers
    params = self.get_regime_parameters(pair)
    # Returns: roi_multiplier, stoploss_multiplier, trailing_multiplier, stake_multiplier
```

---

## Trade Explainability

Attaches structured reasons to each trade for debugging and analysis.

### Configuration

```json
{
    "adaptive_strategy": {
        "explainability": {
            "enabled": true,
            "log_explanations": true,
            "store_explanations": true,
            "include_in_api": true,
            "max_entries_per_trade": 20
        }
    }
}
```

### Usage in Strategy

```python
def confirm_trade_entry(self, pair, order_type, amount, rate, **kwargs):
    # Start building explanation
    explanation = self.start_trade_explanation(pair, 'entry', 'long')
    
    if explanation:
        # Add indicator values
        explanation.add_indicator('RSI', rsi_value, threshold=30, contribution=0.3)
        explanation.add_indicator('MACD', macd_value, contribution=0.2)
        
        # Add market regime
        regime = self.get_market_regime(pair)
        explanation.add_regime(regime.regime.value, regime.confidence)
        
        # Add risk factors
        modulation = self.get_risk_modulation()
        explanation.add_risk_factor('risk_level', modulation.risk_level.value)
        
        # Finalize with confidence score
        self.finalize_trade_explanation(pair, confidence_score=0.75)
    
    return True
```

### Explanation Structure

```json
{
    "trade_id": 123,
    "pair": "BTC/USDT",
    "action": "entry",
    "direction": "long",
    "timestamp": "2024-01-15T10:30:00Z",
    "entries": [
        {
            "category": "indicator",
            "name": "RSI",
            "value": 28.5,
            "threshold": 30,
            "contribution": 0.3,
            "description": "RSI: 28.5 < 30"
        },
        {
            "category": "market_regime",
            "name": "market_regime",
            "value": "trending_up",
            "contribution": 0.2,
            "description": "Market regime: trending_up (confidence: 78%)"
        }
    ],
    "summary": "RSI: 28.5 < 30; Market regime: trending_up (confidence: 78%)",
    "confidence_score": 0.75
}
```

---

## Combined Adaptive Parameters

Get all adaptive adjustments in one call:

```python
def populate_indicators(self, dataframe, metadata):
    pair = metadata['pair']
    
    # Get all adaptive parameters at once
    params = self.get_adaptive_parameters(pair, dataframe)
    
    # Returns:
    # {
    #     'roi_multiplier': 1.2,
    #     'stoploss_multiplier': 1.1,
    #     'stake_multiplier': 0.8,
    #     'trailing_multiplier': 1.0,
    #     'block_entry': False,
    #     'risk_level': 'normal',
    #     'market_regime': 'trending_up',
    #     'regime_confidence': 0.75
    # }
```

---

## Backward Compatibility

All adaptive features are **fully backward compatible**:

- Existing strategies work unchanged
- All features are opt-in
- No changes to default trading behavior
- Strategies not inheriting `AdaptiveStrategyMixin` are unaffected

---

## Example Strategy

See [AdaptiveExampleStrategy](../freqtrade/templates/AdaptiveExampleStrategy.py) for a complete example demonstrating all adaptive features.

---

## Best Practices

1. **Start Conservative** - Enable features one at a time
2. **Backtest Thoroughly** - Use mock signals to test risk filtering
3. **Monitor Profiles** - Check generated profiles match expectations
4. **Log Explanations** - Review trade explanations for debugging
5. **Gradual Rollout** - Test in dry-run before live trading
