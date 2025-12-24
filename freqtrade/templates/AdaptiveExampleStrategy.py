# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
"""
Adaptive Example Strategy

This strategy demonstrates the adaptive strategy features:
- Per-asset parameter profiling
- Contextual risk filtering
- Market regime detection
- Trade explainability

To use adaptive features, add this to your config.json:

{
    "adaptive_strategy": {
        "enabled": true,
        "asset_profiles": {
            "enabled": true,
            "storage_path": "user_data/profiles",
            "update_interval_hours": 24,
            "rolling_window_days": 30,
            "min_trades_for_profile": 10,
            "auto_update": true
        },
        "contextual_risk": {
            "enabled": true,
            "providers": ["mock"],
            "signals_dir": "user_data/signals",
            "extreme_fear_threshold": 20,
            "high_fear_threshold": 35,
            "high_greed_threshold": 75,
            "extreme_greed_threshold": 85,
            "min_stake_multiplier": 0.25,
            "max_stoploss_multiplier": 1.5,
            "fallback_on_error": true
        },
        "market_regime": {
            "enabled": true,
            "lookback_period": 50,
            "adx_trend_threshold": 25.0,
            "adx_strong_trend": 40.0
        },
        "explainability": {
            "enabled": true,
            "log_explanations": true,
            "store_explanations": true,
            "include_in_api": true
        }
    }
}
"""

from datetime import datetime, UTC
from functools import reduce

import numpy as np
import pandas as pd
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.adaptive import AdaptiveStrategyMixin
from freqtrade.persistence import Trade


class AdaptiveExampleStrategy(IStrategy, AdaptiveStrategyMixin):
    """
    Example strategy demonstrating adaptive features.
    
    This strategy:
    - Adjusts stoploss and ROI based on per-asset volatility profiles
    - Reduces position sizes during extreme fear/greed conditions
    - Adapts to market regime (trend vs range)
    - Logs explanations for every trade entry
    """
    
    INTERFACE_VERSION = 3
    
    # Base strategy parameters (will be adjusted per-pair)
    timeframe = '1h'
    
    # Base ROI - will be adjusted by asset profiles
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }
    
    # Base stoploss - will be adjusted by asset profiles
    stoploss = -0.10
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True
    
    # Sell signal
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    # Startup candles needed for indicators
    startup_candle_count: int = 100
    
    # Use custom stoploss for adaptive adjustments
    use_custom_stoploss = True
    
    # Hyperoptable parameters
    buy_rsi = IntParameter(20, 40, default=30, space='buy')
    buy_rsi_regime_trending = IntParameter(25, 50, default=40, space='buy')
    buy_rsi_regime_ranging = IntParameter(15, 35, default=25, space='buy')
    
    sell_rsi = IntParameter(60, 80, default=70, space='sell')
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # Initialize adaptive features
        self.init_adaptive(config)
    
    def bot_start(self, **kwargs) -> None:
        """
        Called once when the bot starts.
        """
        self.log_once("Adaptive Example Strategy started", "info")
        if self.adaptive_enabled:
            self.log_once("Adaptive features are ENABLED", "info")
        else:
            self.log_once("Adaptive features are DISABLED", "info")
    
    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """
        Called at the start of every bot iteration.
        Update external signals here.
        """
        # Update risk signals for contextual risk filtering
        self.update_risk_signals(current_time)
        
        # Log current risk state periodically
        modulation = self.get_risk_modulation()
        if modulation.is_restrictive():
            self.log_once(
                f"Risk modulation active: {', '.join(modulation.reasons)}",
                "warning"
            )
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate all indicators.
        """
        pair = metadata['pair']
        
        # Standard indicators
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        
        # Volatility indicators
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['bb_upper'], dataframe['bb_mid'], dataframe['bb_lower'] = ta.BBANDS(
            dataframe, timeperiod=20, nbdevup=2, nbdevdn=2
        )
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_mid']
        
        # ADX for trend strength
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=14)
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # Update asset profile and market regime if adaptive is enabled
        if self.adaptive_enabled:
            # Update asset profile from this data
            self.update_asset_profile(pair, dataframe)
            
            # Detect market regime
            regime_state = self.detect_market_regime(pair, dataframe)
            
            # Add regime to dataframe for analysis
            dataframe['regime'] = regime_state.regime.value
            dataframe['regime_confidence'] = regime_state.confidence
            
            # Log adaptive state
            self.log_adaptive_state(pair)
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate entry signals based on market conditions.
        """
        pair = metadata['pair']
        
        # Base entry conditions
        conditions = []
        
        # Get adaptive parameters
        adaptive_params = self.get_adaptive_parameters(pair, dataframe)
        
        # Check if entries should be blocked by risk filter
        if adaptive_params.get('block_entry', False):
            # Don't generate any entry signals when blocked
            dataframe.loc[:, 'enter_long'] = 0
            dataframe.loc[:, 'enter_tag'] = ''
            return dataframe
        
        # Adjust RSI threshold based on market regime
        if self.is_trending(pair):
            # In trending market, be less aggressive (higher RSI threshold)
            rsi_threshold = self.buy_rsi_regime_trending.value
            entry_tag_suffix = "_trending"
        elif self.is_ranging(pair):
            # In ranging market, buy dips more aggressively
            rsi_threshold = self.buy_rsi_regime_ranging.value
            entry_tag_suffix = "_ranging"
        else:
            rsi_threshold = self.buy_rsi.value
            entry_tag_suffix = "_normal"
        
        # RSI oversold condition
        conditions.append(dataframe['rsi'] < rsi_threshold)
        
        # Price above EMA 200 (uptrend filter)
        conditions.append(dataframe['close'] > dataframe['ema200'])
        
        # EMA alignment (20 > 50 for uptrend)
        conditions.append(dataframe['ema20'] > dataframe['ema50'])
        
        # MACD positive
        conditions.append(dataframe['macd'] > dataframe['macdsignal'])
        
        # Volume confirmation
        conditions.append(dataframe['volume'] > 0)
        
        # Combine all conditions
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                ['enter_long', 'enter_tag']
            ] = (1, f'rsi_oversold{entry_tag_suffix}')
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate exit signals.
        """
        conditions = []
        
        # RSI overbought
        conditions.append(dataframe['rsi'] > self.sell_rsi.value)
        
        # EMA bearish crossover
        conditions.append(dataframe['ema20'] < dataframe['ema50'])
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                ['exit_long', 'exit_tag']
            ] = (1, 'rsi_overbought')
        
        return dataframe
    
    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs
    ) -> float | None:
        """
        Custom stoploss adjusted by asset profile and risk conditions.
        """
        # Get adaptive stoploss
        adjusted_stoploss = self.get_risk_adjusted_stoploss(pair, self.stoploss)
        
        # Also consider market regime
        regime_params = self.get_regime_parameters(pair)
        if regime_params:
            adjusted_stoploss *= regime_params.get('stoploss_multiplier', 1.0)
        
        # Apply trailing logic manually if in profit
        if current_profit > self.trailing_stop_positive_offset:
            # Use tighter stop when in good profit
            return max(adjusted_stoploss, -current_profit + self.trailing_stop_positive)
        
        return adjusted_stoploss
    
    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs
    ) -> float:
        """
        Adjust stake amount based on asset profile and risk conditions.
        """
        # Get risk-adjusted stake (combines profile + contextual risk)
        adjusted_stake = self.get_risk_adjusted_stake(pair, proposed_stake)
        
        # Ensure we're within bounds
        if min_stake is not None:
            adjusted_stake = max(min_stake, adjusted_stake)
        adjusted_stake = min(max_stake, adjusted_stake)
        
        return adjusted_stake
    
    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs
    ) -> bool:
        """
        Confirm trade entry and build explanation.
        """
        # Start building explanation
        explanation = self.start_trade_explanation(pair, 'entry', side)
        
        if explanation is not None:
            # Get current dataframe for indicator values
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                last_row = dataframe.iloc[-1]
                
                # Add indicator explanations
                explanation.add_indicator(
                    'RSI', 
                    float(last_row.get('rsi', 0)),
                    threshold=self.buy_rsi.value,
                    contribution=0.3,
                    description=f"RSI: {last_row.get('rsi', 0):.1f} < {self.buy_rsi.value}"
                )
                
                explanation.add_indicator(
                    'MACD',
                    float(last_row.get('macdhist', 0)),
                    contribution=0.2,
                    description=f"MACD histogram: {last_row.get('macdhist', 0):.4f}"
                )
            
            # Add market regime
            regime_state = self.get_market_regime(pair)
            explanation.add_regime(
                regime_state.regime.value,
                regime_state.confidence,
                contribution=0.2
            )
            
            # Add risk factors
            modulation = self.get_risk_modulation()
            if modulation.is_restrictive():
                explanation.add_risk_factor(
                    'risk_modulation',
                    modulation.risk_level.value,
                    contribution=-0.1,
                    description=f"Risk level: {modulation.risk_level.value}"
                )
            
            # Add profile factors
            profile = self.get_asset_profile(pair)
            if profile:
                explanation.add_profile_factor(
                    'volatility',
                    profile.volatility,
                    contribution=0.1,
                    description=f"Asset volatility: {profile.volatility:.4f}"
                )
                explanation.add_profile_factor(
                    'win_rate',
                    profile.win_loss_ratio,
                    contribution=0.1,
                    description=f"Historical W/L ratio: {profile.win_loss_ratio:.2f}"
                )
            
            # Calculate confidence
            confidence = min(1.0, max(0.0, 
                0.5 + sum(e.contribution for e in explanation.entries)
            ))
            
            # Finalize explanation
            self.finalize_trade_explanation(pair, confidence_score=confidence)
        
        # Check if entries are blocked by risk filter
        if self.should_block_entry():
            self.log_once(f"Entry blocked for {pair} due to risk conditions", "warning")
            self.discard_explanation(pair)
            return False
        
        return True
    
    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs
    ) -> bool:
        """
        Confirm trade exit and log explanation.
        """
        # Build exit explanation
        explanation = self.start_trade_explanation(pair, 'exit', trade.trade_direction)
        
        if explanation is not None:
            explanation.add_signal(
                'exit_reason',
                exit_reason,
                contribution=0.5,
                description=f"Exit reason: {exit_reason}"
            )
            
            # Add profit info
            current_profit = trade.calc_profit_ratio(rate)
            explanation.metadata['profit_ratio'] = current_profit
            explanation.metadata['trade_duration'] = (
                current_time - trade.open_date
            ).total_seconds() / 60
            
            self.finalize_trade_explanation(pair, trade.id, confidence_score=0.8)
        
        return True
