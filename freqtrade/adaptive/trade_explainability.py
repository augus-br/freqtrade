"""
Trade Explainability Module

Attaches structured reasons to each trade (indicator + context)
for debugging and analysis purposes.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Any

from freqtrade.constants import Config


logger = logging.getLogger(__name__)


class ExplanationCategory(str, Enum):
    """Categories for trade explanations."""
    INDICATOR = "indicator"
    SIGNAL = "signal"
    RISK_FILTER = "risk_filter"
    ASSET_PROFILE = "asset_profile"
    MARKET_REGIME = "market_regime"
    USER_DEFINED = "user_defined"


@dataclass
class ExplanationEntry:
    """
    A single explanation entry for a trade decision.
    
    Attributes:
        category: Category of the explanation
        name: Name of the factor (e.g., 'RSI', 'fear_greed')
        value: Value at the time of decision
        threshold: Threshold that was compared against (if applicable)
        contribution: Contribution to the decision (-1.0 to 1.0)
        description: Human-readable description
    """
    category: ExplanationCategory
    name: str
    value: float | str | None = None
    threshold: float | str | None = None
    contribution: float = 0.0
    description: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "name": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "contribution": self.contribution,
            "description": self.description,
        }


@dataclass
class TradeExplanation:
    """
    Complete explanation for a trade entry or exit.
    
    Attributes:
        trade_id: Associated trade ID (if available)
        pair: Trading pair
        action: 'entry' or 'exit'
        direction: 'long' or 'short'
        timestamp: When the explanation was generated
        entries: List of explanation entries
        summary: Brief summary of the main reasons
        confidence_score: Overall confidence in the trade (0.0 to 1.0)
        metadata: Additional contextual information
    """
    pair: str
    action: str  # 'entry' or 'exit'
    direction: str  # 'long' or 'short'
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    trade_id: int | None = None
    entries: list[ExplanationEntry] = field(default_factory=list)
    summary: str = ""
    confidence_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def add_indicator(
        self,
        name: str,
        value: float,
        threshold: float | None = None,
        contribution: float = 0.0,
        description: str = ""
    ) -> None:
        """Add an indicator-based explanation entry."""
        self.entries.append(ExplanationEntry(
            category=ExplanationCategory.INDICATOR,
            name=name,
            value=value,
            threshold=threshold,
            contribution=contribution,
            description=description or f"{name}: {value}"
        ))
    
    def add_signal(
        self,
        name: str,
        value: Any,
        contribution: float = 0.0,
        description: str = ""
    ) -> None:
        """Add a signal-based explanation entry."""
        self.entries.append(ExplanationEntry(
            category=ExplanationCategory.SIGNAL,
            name=name,
            value=str(value),
            contribution=contribution,
            description=description
        ))
    
    def add_risk_factor(
        self,
        name: str,
        value: Any,
        contribution: float = 0.0,
        description: str = ""
    ) -> None:
        """Add a risk filter explanation entry."""
        self.entries.append(ExplanationEntry(
            category=ExplanationCategory.RISK_FILTER,
            name=name,
            value=str(value) if not isinstance(value, (int, float)) else value,
            contribution=contribution,
            description=description
        ))
    
    def add_regime(
        self,
        regime: str,
        confidence: float,
        contribution: float = 0.0
    ) -> None:
        """Add market regime explanation."""
        self.entries.append(ExplanationEntry(
            category=ExplanationCategory.MARKET_REGIME,
            name="market_regime",
            value=regime,
            threshold=None,
            contribution=contribution,
            description=f"Market regime: {regime} (confidence: {confidence:.1%})"
        ))
    
    def add_profile_factor(
        self,
        name: str,
        value: float,
        contribution: float = 0.0,
        description: str = ""
    ) -> None:
        """Add asset profile explanation."""
        self.entries.append(ExplanationEntry(
            category=ExplanationCategory.ASSET_PROFILE,
            name=name,
            value=value,
            contribution=contribution,
            description=description or f"Profile {name}: {value:.4f}"
        ))
    
    def generate_summary(self) -> str:
        """Generate a brief summary from entries."""
        if not self.entries:
            return "No explanation entries"
        
        # Sort by contribution magnitude
        sorted_entries = sorted(
            self.entries,
            key=lambda e: abs(e.contribution),
            reverse=True
        )[:3]  # Top 3 factors
        
        factors = []
        for entry in sorted_entries:
            if entry.description:
                factors.append(entry.description)
            else:
                factors.append(f"{entry.name}={entry.value}")
        
        self.summary = "; ".join(factors)
        return self.summary
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/API."""
        return {
            "trade_id": self.trade_id,
            "pair": self.pair,
            "action": self.action,
            "direction": self.direction,
            "timestamp": self.timestamp.isoformat(),
            "entries": [e.to_dict() for e in self.entries],
            "summary": self.summary or self.generate_summary(),
            "confidence_score": self.confidence_score,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TradeExplanation":
        """Create from dictionary."""
        exp = cls(
            pair=data["pair"],
            action=data["action"],
            direction=data["direction"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            trade_id=data.get("trade_id"),
            summary=data.get("summary", ""),
            confidence_score=data.get("confidence_score", 0.0),
            metadata=data.get("metadata", {}),
        )
        for entry_data in data.get("entries", []):
            exp.entries.append(ExplanationEntry(
                category=ExplanationCategory(entry_data["category"]),
                name=entry_data["name"],
                value=entry_data.get("value"),
                threshold=entry_data.get("threshold"),
                contribution=entry_data.get("contribution", 0.0),
                description=entry_data.get("description", ""),
            ))
        return exp


@dataclass
class ExplainabilityConfig:
    """Configuration for trade explainability."""
    enabled: bool = False
    log_explanations: bool = True
    store_explanations: bool = True
    include_in_api: bool = True
    max_entries_per_trade: int = 20
    
    @classmethod
    def from_config(cls, config: Config) -> "ExplainabilityConfig":
        """Create from Freqtrade config."""
        adaptive_config = config.get("adaptive_strategy", {})
        explain_config = adaptive_config.get("explainability", {})
        return cls(
            enabled=explain_config.get("enabled", False),
            log_explanations=explain_config.get("log_explanations", True),
            store_explanations=explain_config.get("store_explanations", True),
            include_in_api=explain_config.get("include_in_api", True),
            max_entries_per_trade=explain_config.get("max_entries_per_trade", 20),
        )


class TradeExplainabilityManager:
    """
    Manages trade explanations for debugging and analysis.
    
    Collects indicator values, risk factors, and contextual information
    at the time of trade decisions for later review.
    """
    
    def __init__(self, config: Config) -> None:
        self.config = config
        self.explain_config = ExplainabilityConfig.from_config(config)
        self._pending_explanations: dict[str, TradeExplanation] = {}  # pair -> explanation
        self._trade_explanations: dict[int, TradeExplanation] = {}    # trade_id -> explanation
    
    @property
    def enabled(self) -> bool:
        """Check if explainability is enabled."""
        return self.explain_config.enabled
    
    def start_explanation(
        self,
        pair: str,
        action: str,
        direction: str = "long"
    ) -> TradeExplanation:
        """
        Start building an explanation for a potential trade.
        
        Args:
            pair: Trading pair
            action: 'entry' or 'exit'
            direction: 'long' or 'short'
        
        Returns:
            New TradeExplanation object to populate
        """
        explanation = TradeExplanation(
            pair=pair,
            action=action,
            direction=direction,
        )
        self._pending_explanations[pair] = explanation
        return explanation
    
    def get_pending(self, pair: str) -> TradeExplanation | None:
        """Get pending explanation for a pair."""
        return self._pending_explanations.get(pair)
    
    def finalize_explanation(
        self,
        pair: str,
        trade_id: int | None = None,
        confidence_score: float = 0.0
    ) -> TradeExplanation | None:
        """
        Finalize and store an explanation.
        
        Args:
            pair: Trading pair
            trade_id: Associated trade ID (if trade was executed)
            confidence_score: Overall confidence in the trade
        
        Returns:
            Finalized explanation, or None if no pending explanation
        """
        explanation = self._pending_explanations.pop(pair, None)
        if explanation is None:
            return None
        
        explanation.trade_id = trade_id
        explanation.confidence_score = confidence_score
        explanation.generate_summary()
        
        # Limit entries
        if len(explanation.entries) > self.explain_config.max_entries_per_trade:
            # Sort by contribution and keep top entries
            explanation.entries.sort(key=lambda e: abs(e.contribution), reverse=True)
            explanation.entries = explanation.entries[:self.explain_config.max_entries_per_trade]
        
        # Store by trade ID if available
        if trade_id is not None:
            self._trade_explanations[trade_id] = explanation
        
        # Log if enabled
        if self.explain_config.log_explanations:
            logger.info(
                f"Trade explanation [{pair}] {explanation.action}: {explanation.summary}"
            )
        
        return explanation
    
    def discard_pending(self, pair: str) -> None:
        """Discard pending explanation for a pair (trade not executed)."""
        self._pending_explanations.pop(pair, None)
    
    def get_explanation(self, trade_id: int) -> TradeExplanation | None:
        """Get stored explanation for a trade."""
        return self._trade_explanations.get(trade_id)
    
    def get_all_explanations(self) -> dict[int, TradeExplanation]:
        """Get all stored explanations."""
        return self._trade_explanations.copy()
    
    def clear_old_explanations(self, keep_trade_ids: list[int]) -> int:
        """
        Clear explanations for closed trades.
        
        Args:
            keep_trade_ids: List of trade IDs to keep
        
        Returns:
            Number of explanations removed
        """
        to_remove = [tid for tid in self._trade_explanations if tid not in keep_trade_ids]
        for tid in to_remove:
            del self._trade_explanations[tid]
        return len(to_remove)


@dataclass
class StrategyConfidenceScore:
    """
    Strategy confidence score based on cross-pair stability.
    
    Quantifies robustness based on:
    - Win rate stability across pairs
    - Drawdown sensitivity
    - Parameter stability
    """
    overall_score: float = 0.0
    stability_score: float = 0.0
    consistency_score: float = 0.0
    risk_adjusted_score: float = 0.0
    pair_scores: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "stability_score": self.stability_score,
            "consistency_score": self.consistency_score,
            "risk_adjusted_score": self.risk_adjusted_score,
            "pair_scores": self.pair_scores,
            "warnings": self.warnings,
        }


class ConfidenceScoreCalculator:
    """
    Calculates strategy confidence scores.
    
    Used to detect overfitting and unstable parameter sets.
    """
    
    def __init__(self, config: Config) -> None:
        self.config = config
    
    def calculate_score(
        self,
        pair_metrics: dict[str, dict[str, float]]
    ) -> StrategyConfidenceScore:
        """
        Calculate confidence score from per-pair metrics.
        
        Args:
            pair_metrics: Dict mapping pair to metrics dict with:
                - 'win_rate': Win rate for the pair
                - 'profit_factor': Profit factor
                - 'max_drawdown': Maximum drawdown
                - 'trade_count': Number of trades
        
        Returns:
            StrategyConfidenceScore
        """
        score = StrategyConfidenceScore()
        
        if not pair_metrics:
            return score
        
        win_rates = []
        profit_factors = []
        drawdowns = []
        
        for pair, metrics in pair_metrics.items():
            wr = metrics.get("win_rate", 0.5)
            pf = metrics.get("profit_factor", 1.0)
            dd = metrics.get("max_drawdown", 0.0)
            
            win_rates.append(wr)
            profit_factors.append(pf)
            drawdowns.append(dd)
            
            # Individual pair score
            pair_score = self._calculate_pair_score(wr, pf, dd)
            score.pair_scores[pair] = pair_score
        
        import numpy as np
        
        # Stability score (low variance = high stability)
        wr_std = np.std(win_rates) if len(win_rates) > 1 else 0
        score.stability_score = max(0, 1.0 - wr_std * 5)
        
        # Consistency score (all pairs profitable)
        profitable_pairs = sum(1 for pf in profit_factors if pf > 1.0)
        score.consistency_score = profitable_pairs / len(profit_factors)
        
        # Risk-adjusted score
        avg_wr = np.mean(win_rates)
        avg_dd = np.mean(drawdowns)
        score.risk_adjusted_score = avg_wr * (1.0 - avg_dd)
        
        # Overall score
        score.overall_score = (
            score.stability_score * 0.3 +
            score.consistency_score * 0.4 +
            score.risk_adjusted_score * 0.3
        )
        
        # Generate warnings
        score.warnings = self._generate_warnings(
            win_rates, profit_factors, drawdowns, pair_metrics
        )
        
        return score
    
    def _calculate_pair_score(
        self,
        win_rate: float,
        profit_factor: float,
        max_drawdown: float
    ) -> float:
        """Calculate confidence score for a single pair."""
        # Win rate component (0-40 points)
        wr_score = min(40, win_rate * 80)
        
        # Profit factor component (0-30 points)
        pf_score = min(30, (profit_factor - 1.0) * 15) if profit_factor > 1 else 0
        
        # Drawdown penalty (0-30 points deducted)
        dd_penalty = min(30, max_drawdown * 100)
        
        return (wr_score + pf_score - dd_penalty) / 100
    
    def _generate_warnings(
        self,
        win_rates: list[float],
        profit_factors: list[float],
        drawdowns: list[float],
        pair_metrics: dict[str, dict[str, float]]
    ) -> list[str]:
        """Generate warnings about potential issues."""
        import numpy as np
        
        warnings = []
        
        # High variance warning
        if np.std(win_rates) > 0.15:
            warnings.append("High variance in win rates across pairs - possible overfitting")
        
        # Low trade count warning
        for pair, metrics in pair_metrics.items():
            if metrics.get("trade_count", 0) < 30:
                warnings.append(f"Low trade count for {pair} - results may not be statistically significant")
        
        # High drawdown warning
        if np.max(drawdowns) > 0.3:
            warnings.append("High maximum drawdown detected - strategy may have high risk")
        
        # Inconsistent profitability warning
        losing_pairs = sum(1 for pf in profit_factors if pf < 1.0)
        if losing_pairs > len(profit_factors) * 0.3:
            warnings.append(f"{losing_pairs} pairs are not profitable - review strategy parameters")
        
        return warnings
