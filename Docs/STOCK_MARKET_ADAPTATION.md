# Stock Market Adaptation Plan - TFT for NASDAQ Trading

**Date**: 2025-10-14
**Status**: Planning Phase
**Target**: Convert server monitoring system to stock prediction system

---

## Executive Summary

This document outlines the conversion plan to adapt our **Temporal Fusion Transformer (TFT)** server monitoring system into a **NASDAQ stock trading prediction system** with sentiment analysis.

### Key Advantages of Our Architecture

✅ **Multi-horizon predictions** - Predict 5min, 1hr, 1day, 1week simultaneously
✅ **Uncertainty bands** (P10/P50/P90) - Perfect for stop-loss and take-profit strategies
✅ **Attention mechanism** - Learns when sentiment matters vs technical indicators
✅ **Transfer learning** - Sector-based profiles (Tech, Finance, Healthcare, etc.)
✅ **Centralized config** - Add/remove features with one-line changes
✅ **Production-ready** - Real-time inference, dashboard, monitoring

### Trading Strategy

**Two-Track Approach**:
1. **Long-term holds** - Identify strong stocks for 1-4 week positions
2. **Day/Week trades** - Capture short-term momentum (1-5 day swings)

**Target Universe**: NASDAQ top 100 stocks (refreshed weekly)

**Unique Edge**: Real-time sentiment analysis using LM Studio with local models

---

## System Architecture Overview

### Current System (Server Monitoring)
```
Server Metrics → TFT Model → Predictions → Dashboard → Alerts
     ↓              ↓            ↓            ↓          ↓
  14 metrics    Profiles    Multi-horizon  Streamlit  Risk scores
```

### Future System (Stock Trading)
```
Stock Data + Sentiment → TFT Model → Buy/Sell Signals → Dashboard → Alerts
        ↓                    ↓              ↓              ↓          ↓
  20+ features          Sectors      Multi-horizon    Streamlit   Trade recs
```

---

## Phase 1: Foundation & Data Pipeline (Week 1-2)

### 1.1 Project Setup

**Action**: Create new project from clean copy
```bash
# Clone current system
git clone https://github.com/Deverydoo/MonitoringPrediction StockTFT
cd StockTFT

# Clean out server-specific files (keep architecture)
# Keep: TFT trainer, inference daemon, dashboard framework
# Remove: Server metrics generator, server profiles, LINBORG schema
```

**Files to Keep (Core Architecture)**:
- ✅ `tft_trainer.py` - TFT training logic (95% reusable)
- ✅ `tft_inference_daemon.py` - Real-time predictions (90% reusable)
- ✅ `tft_dashboard_web.py` - Dashboard framework (80% reusable)
- ✅ `end_to_end_certification.py` - Testing framework (adapt tests)
- ✅ `main.py` - CLI interface (keep structure)

**Files to Replace (Domain-Specific)**:
- ❌ `linborg_schema.py` → `stock_features.py`
- ❌ `server_profiles.py` → `sector_profiles.py`
- ❌ `metrics_generator.py` → `stock_data_fetcher.py`

### 1.2 Define Stock Feature Schema

**Create**: `stock_features.py` (replaces `linborg_schema.py`)

```python
# -*- coding: utf-8 -*-
"""
Stock Market Feature Schema
Single source of truth for all stock features used in TFT model
"""

from typing import List, Dict, Tuple
from enum import Enum

# ============================================================================
# CORE STOCK FEATURES (20 features)
# ============================================================================

STOCK_FEATURES = [
    # === Price Features (5) ===
    'open',
    'high',
    'low',
    'close',
    'volume',

    # === Technical Indicators (8) ===
    'RSI',                    # Relative Strength Index (14-day)
    'MACD',                   # Moving Average Convergence Divergence
    'MACD_signal',            # MACD signal line
    'bollinger_upper',        # Bollinger Band upper
    'bollinger_lower',        # Bollinger Band lower
    'SMA_20',                 # Simple Moving Average 20-day
    'SMA_50',                 # Simple Moving Average 50-day
    'ATR',                    # Average True Range (volatility)

    # === Sentiment Features (4) ===
    'sentiment_score',        # 0-100 from news analysis
    'sentiment_momentum',     # Rate of sentiment change
    'sentiment_volatility',   # Sentiment stability
    'news_volume',            # Number of articles (last 24h)

    # === Market Context (3) ===
    'VIX',                    # Market volatility index
    'sector_performance',     # Sector's daily performance %
    'market_correlation',     # Correlation with S&P500
]

NUM_STOCK_FEATURES = len(STOCK_FEATURES)

# ============================================================================
# FEATURE CATEGORIZATION
# ============================================================================

PRICE_FEATURES = [
    'open', 'high', 'low', 'close', 'volume'
]

TECHNICAL_FEATURES = [
    'RSI', 'MACD', 'MACD_signal', 'bollinger_upper', 'bollinger_lower',
    'SMA_20', 'SMA_50', 'ATR'
]

SENTIMENT_FEATURES = [
    'sentiment_score', 'sentiment_momentum', 'sentiment_volatility', 'news_volume'
]

MARKET_CONTEXT_FEATURES = [
    'VIX', 'sector_performance', 'market_correlation'
]

# Critical features for alerting (if missing, warn)
CRITICAL_FEATURES = [
    'close', 'volume', 'sentiment_score', 'RSI'
]

# ============================================================================
# PREDICTION TARGETS
# ============================================================================

PREDICTION_TARGETS = [
    'close',              # Primary target: predict future close price
    'high',               # Secondary: predict future high (for take-profit)
    'low',                # Secondary: predict future low (for stop-loss)
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_stock_features(df_columns: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate that dataframe has required stock features.

    Args:
        df_columns: List of column names from DataFrame

    Returns:
        Tuple of (present_features, missing_features)
    """
    present = [f for f in STOCK_FEATURES if f in df_columns]
    missing = [f for f in STOCK_FEATURES if f not in df_columns]
    return present, missing

def get_feature_type(feature_name: str) -> str:
    """
    Determine feature type for normalization.

    Returns:
        'price', 'percentage', 'volume', 'sentiment', 'indicator', 'unknown'
    """
    if feature_name in ['open', 'high', 'low', 'close']:
        return 'price'
    elif feature_name in ['volume']:
        return 'volume'
    elif feature_name in SENTIMENT_FEATURES:
        return 'sentiment'
    elif feature_name in ['sector_performance', 'market_correlation']:
        return 'percentage'
    elif feature_name in TECHNICAL_FEATURES:
        return 'indicator'
    else:
        return 'unknown'

def get_feature_description(feature_name: str) -> str:
    """Get human-readable description of feature."""
    descriptions = {
        'sentiment_score': 'News sentiment analysis (0=bearish, 100=bullish)',
        'sentiment_momentum': 'Rate of sentiment change (positive=improving)',
        'sentiment_volatility': 'Sentiment stability (low=stable, high=volatile)',
        'news_volume': 'Number of news articles in last 24h',
        'VIX': 'Market fear index (low=calm, high=volatile)',
        'RSI': 'Relative Strength Index (>70=overbought, <30=oversold)',
        'MACD': 'Trend strength indicator',
        'ATR': 'Average True Range (volatility measure)',
    }
    return descriptions.get(feature_name, feature_name)

# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'STOCK_FEATURES',
    'NUM_STOCK_FEATURES',
    'PRICE_FEATURES',
    'TECHNICAL_FEATURES',
    'SENTIMENT_FEATURES',
    'MARKET_CONTEXT_FEATURES',
    'CRITICAL_FEATURES',
    'PREDICTION_TARGETS',
    'validate_stock_features',
    'get_feature_type',
    'get_feature_description',
]
```

### 1.3 Define Sector Profiles

**Create**: `sector_profiles.py` (replaces `server_profiles.py`)

```python
# -*- coding: utf-8 -*-
"""
NASDAQ Sector Profiles
Centralized sector detection and baseline characteristics
"""

import re
from enum import Enum
from typing import Dict, List, Tuple

# ============================================================================
# SECTOR ENUMERATION
# ============================================================================

class SectorProfile(Enum):
    """NASDAQ sector classifications"""
    TECHNOLOGY = "technology"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    COMMUNICATION = "communication"
    INDUSTRIALS = "industrials"
    ENERGY = "energy"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"
    MATERIALS = "materials"
    UNKNOWN = "unknown"

# ============================================================================
# PRICE TIER CLASSIFICATION
# ============================================================================

class PriceTier(Enum):
    """
    Stock price tiers for different trading strategies.

    Different price ranges have different characteristics:
    - PENNY: High volatility, high risk, potential for large % gains
    - LOW: Good for day trading, lower capital requirements
    - MID: Balanced risk/reward
    - HIGH: More stable, institutional interest
    - MEGA: Blue chips, lower volatility, long-term holds
    """
    PENNY = "penny"           # $0.50 - $5.00
    LOW = "low"               # $5.01 - $25.00
    MID = "mid"               # $25.01 - $100.00
    HIGH = "high"             # $100.01 - $500.00
    MEGA = "mega"             # $500.00+

# Price tier boundaries (configurable)
PRICE_TIER_RANGES = {
    PriceTier.PENNY: (0.50, 5.00),
    PriceTier.LOW: (5.01, 25.00),
    PriceTier.MID: (25.01, 100.00),
    PriceTier.HIGH: (100.01, 500.00),
    PriceTier.MEGA: (500.01, float('inf'))
}

# Trading characteristics by price tier
PRICE_TIER_CHARACTERISTICS = {
    PriceTier.PENNY: {
        'avg_daily_volatility': 5.0,    # % daily price swing
        'sentiment_weight': 2.0,        # Very sensitive to news
        'min_volume': 500_000,          # Liquidity requirement
        'position_size_limit': 0.05,    # Max 5% of portfolio (risky)
        'stop_loss_pct': 0.10,          # 10% stop-loss (high risk)
        'ideal_horizon': 'day_trade',   # Quick in/out
    },
    PriceTier.LOW: {
        'avg_daily_volatility': 3.0,
        'sentiment_weight': 1.5,
        'min_volume': 1_000_000,
        'position_size_limit': 0.08,    # Max 8% of portfolio
        'stop_loss_pct': 0.05,          # 5% stop-loss
        'ideal_horizon': 'day_trade',
    },
    PriceTier.MID: {
        'avg_daily_volatility': 2.0,
        'sentiment_weight': 1.2,
        'min_volume': 2_000_000,
        'position_size_limit': 0.10,    # Max 10% of portfolio
        'stop_loss_pct': 0.03,          # 3% stop-loss
        'ideal_horizon': 'swing_trade',
    },
    PriceTier.HIGH: {
        'avg_daily_volatility': 1.5,
        'sentiment_weight': 1.0,
        'min_volume': 3_000_000,
        'position_size_limit': 0.12,    # Max 12% of portfolio
        'stop_loss_pct': 0.02,          # 2% stop-loss
        'ideal_horizon': 'position',
    },
    PriceTier.MEGA: {
        'avg_daily_volatility': 1.0,
        'sentiment_weight': 0.8,
        'min_volume': 5_000_000,
        'position_size_limit': 0.15,    # Max 15% of portfolio (safer)
        'stop_loss_pct': 0.02,          # 2% stop-loss
        'ideal_horizon': 'position',
    },
}

# ============================================================================
# TICKER-TO-SECTOR MAPPING (Top NASDAQ 100)
# ============================================================================

TICKER_SECTOR_MAP = {
    # Technology (Mega-cap)
    'AAPL': SectorProfile.TECHNOLOGY,
    'MSFT': SectorProfile.TECHNOLOGY,
    'GOOGL': SectorProfile.TECHNOLOGY,
    'GOOG': SectorProfile.TECHNOLOGY,
    'NVDA': SectorProfile.TECHNOLOGY,
    'META': SectorProfile.TECHNOLOGY,
    'TSLA': SectorProfile.TECHNOLOGY,  # Auto + Tech
    'AVGO': SectorProfile.TECHNOLOGY,
    'ADBE': SectorProfile.TECHNOLOGY,
    'CSCO': SectorProfile.TECHNOLOGY,
    'INTC': SectorProfile.TECHNOLOGY,
    'AMD': SectorProfile.TECHNOLOGY,
    'QCOM': SectorProfile.TECHNOLOGY,
    'TXN': SectorProfile.TECHNOLOGY,
    'AMAT': SectorProfile.TECHNOLOGY,

    # Healthcare
    'AMGN': SectorProfile.HEALTHCARE,
    'GILD': SectorProfile.HEALTHCARE,
    'VRTX': SectorProfile.HEALTHCARE,
    'REGN': SectorProfile.HEALTHCARE,
    'BIIB': SectorProfile.HEALTHCARE,

    # Consumer Discretionary
    'AMZN': SectorProfile.CONSUMER_DISCRETIONARY,
    'COST': SectorProfile.CONSUMER_DISCRETIONARY,
    'SBUX': SectorProfile.CONSUMER_DISCRETIONARY,
    'BKNG': SectorProfile.CONSUMER_DISCRETIONARY,

    # Consumer Staples
    'PEP': SectorProfile.CONSUMER_STAPLES,
    'MDLZ': SectorProfile.CONSUMER_STAPLES,

    # Communication Services
    'NFLX': SectorProfile.COMMUNICATION,
    'CMCSA': SectorProfile.COMMUNICATION,
    'CHTR': SectorProfile.COMMUNICATION,

    # Finance
    'PYPL': SectorProfile.FINANCE,

    # Add more as needed...
}

# ============================================================================
# SECTOR CHARACTERISTICS (Baseline Patterns)
# ============================================================================

SECTOR_BASELINES = {
    SectorProfile.TECHNOLOGY: {
        'avg_volatility': 2.5,        # % daily price movement
        'avg_volume': 25_000_000,     # Typical daily volume
        'sentiment_weight': 1.5,      # How much sentiment matters (1.0 = normal)
        'news_frequency': 50,         # Articles per week
        'typical_RSI': 55,            # Tends to run hot
        'beta': 1.3,                  # More volatile than market
    },
    SectorProfile.FINANCE: {
        'avg_volatility': 1.8,
        'avg_volume': 15_000_000,
        'sentiment_weight': 1.2,
        'news_frequency': 30,
        'typical_RSI': 50,
        'beta': 1.1,
    },
    SectorProfile.HEALTHCARE: {
        'avg_volatility': 2.0,
        'avg_volume': 8_000_000,
        'sentiment_weight': 1.8,      # Clinical trials = high sentiment impact
        'news_frequency': 40,
        'typical_RSI': 52,
        'beta': 0.9,
    },
    SectorProfile.CONSUMER_DISCRETIONARY: {
        'avg_volatility': 2.2,
        'avg_volume': 12_000_000,
        'sentiment_weight': 1.3,
        'news_frequency': 35,
        'typical_RSI': 53,
        'beta': 1.2,
    },
    SectorProfile.CONSUMER_STAPLES: {
        'avg_volatility': 1.2,
        'avg_volume': 10_000_000,
        'sentiment_weight': 0.8,      # Less sensitive to news
        'news_frequency': 20,
        'typical_RSI': 50,
        'beta': 0.7,                  # Defensive sector
    },
    SectorProfile.COMMUNICATION: {
        'avg_volatility': 2.3,
        'avg_volume': 18_000_000,
        'sentiment_weight': 1.4,
        'news_frequency': 45,
        'typical_RSI': 52,
        'beta': 1.1,
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def infer_sector_from_ticker(ticker: str) -> SectorProfile:
    """
    Infer sector from ticker symbol.

    Args:
        ticker: Stock ticker (e.g., 'AAPL', 'MSFT')

    Returns:
        SectorProfile enum
    """
    ticker = ticker.upper().strip()
    return TICKER_SECTOR_MAP.get(ticker, SectorProfile.UNKNOWN)

def get_sector_display_name(sector: SectorProfile) -> str:
    """
    Get human-friendly sector name.

    Args:
        sector: SectorProfile enum

    Returns:
        Display name (e.g., "Technology")
    """
    display_names = {
        SectorProfile.TECHNOLOGY: "Technology",
        SectorProfile.FINANCE: "Finance",
        SectorProfile.HEALTHCARE: "Healthcare",
        SectorProfile.CONSUMER_DISCRETIONARY: "Consumer Discretionary",
        SectorProfile.CONSUMER_STAPLES: "Consumer Staples",
        SectorProfile.COMMUNICATION: "Communication Services",
        SectorProfile.INDUSTRIALS: "Industrials",
        SectorProfile.ENERGY: "Energy",
        SectorProfile.UTILITIES: "Utilities",
        SectorProfile.REAL_ESTATE: "Real Estate",
        SectorProfile.MATERIALS: "Materials",
        SectorProfile.UNKNOWN: "Unknown",
    }
    return display_names.get(sector, sector.value)

def get_sector_baselines(sector: SectorProfile) -> Dict:
    """
    Get baseline characteristics for sector.

    Args:
        sector: SectorProfile enum

    Returns:
        Dictionary of baseline metrics
    """
    return SECTOR_BASELINES.get(sector, {
        'avg_volatility': 2.0,
        'avg_volume': 10_000_000,
        'sentiment_weight': 1.0,
        'news_frequency': 30,
        'typical_RSI': 50,
        'beta': 1.0,
    })

def get_all_tickers_for_sector(sector: SectorProfile) -> List[str]:
    """Get all tickers in a specific sector."""
    return [ticker for ticker, s in TICKER_SECTOR_MAP.items() if s == sector]

def infer_price_tier(price: float) -> PriceTier:
    """
    Determine price tier from current stock price.

    Args:
        price: Current stock price

    Returns:
        PriceTier enum
    """
    for tier, (min_price, max_price) in PRICE_TIER_RANGES.items():
        if min_price <= price <= max_price:
            return tier
    return PriceTier.MID  # Default fallback

def get_price_tier_characteristics(tier: PriceTier) -> Dict:
    """
    Get trading characteristics for price tier.

    Args:
        tier: PriceTier enum

    Returns:
        Dictionary of characteristics (volatility, sentiment weight, etc.)
    """
    return PRICE_TIER_CHARACTERISTICS.get(tier, PRICE_TIER_CHARACTERISTICS[PriceTier.MID])

def filter_by_price_tier(
    tickers: List[str],
    prices: Dict[str, float],
    allowed_tiers: List[PriceTier]
) -> List[str]:
    """
    Filter tickers by price tier.

    Args:
        tickers: List of stock tickers
        prices: Dict of {ticker: current_price}
        allowed_tiers: List of allowed PriceTier enums

    Returns:
        Filtered list of tickers in allowed price ranges
    """
    filtered = []
    for ticker in tickers:
        if ticker not in prices:
            continue
        tier = infer_price_tier(prices[ticker])
        if tier in allowed_tiers:
            filtered.append(ticker)
    return filtered

# ============================================================================
# NASDAQ 100 TRACKING
# ============================================================================

def get_nasdaq_100_tickers() -> List[str]:
    """
    Get list of NASDAQ 100 tickers.
    NOTE: This should be refreshed weekly from actual NASDAQ 100 index.
    """
    # Return all tickers we have mapped
    return list(TICKER_SECTOR_MAP.keys())

def refresh_nasdaq_100(new_tickers: Dict[str, SectorProfile]):
    """
    Update NASDAQ 100 tracking list (called weekly).

    Args:
        new_tickers: Dict of {ticker: sector} for current NASDAQ 100
    """
    global TICKER_SECTOR_MAP
    TICKER_SECTOR_MAP = new_tickers

# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'SectorProfile',
    'PriceTier',
    'TICKER_SECTOR_MAP',
    'SECTOR_BASELINES',
    'PRICE_TIER_RANGES',
    'PRICE_TIER_CHARACTERISTICS',
    'infer_sector_from_ticker',
    'get_sector_display_name',
    'get_sector_baselines',
    'get_all_tickers_for_sector',
    'infer_price_tier',
    'get_price_tier_characteristics',
    'filter_by_price_tier',
    'get_nasdaq_100_tickers',
    'refresh_nasdaq_100',
]
```

### 1.4 Price Tier Configuration & Filtering

**Why Price Tiers Matter**:

Different price ranges have fundamentally different trading characteristics:

| Price Tier | Range | Volatility | Best For | Risk Level |
|------------|-------|------------|----------|------------|
| **Penny** | $0.50-$5 | Very High (5%) | Day trades, high risk/reward | 🔴 HIGH |
| **Low** | $5-$25 | High (3%) | Day/swing trades | 🟡 MEDIUM-HIGH |
| **Mid** | $25-$100 | Medium (2%) | Swing trades, balanced | 🟢 MEDIUM |
| **High** | $100-$500 | Low (1.5%) | Position trades, stable | 🔵 LOW |
| **Mega** | $500+ | Very Low (1%) | Long-term holds, blue chips | 🟢 VERY LOW |

**Configuration Example**:

```python
from sector_profiles import PriceTier, filter_by_price_tier, infer_price_tier

# Configure which price tiers to trade
# Option 1: Focus on low-price day trading
ALLOWED_TIERS = [PriceTier.PENNY, PriceTier.LOW]

# Option 2: Balanced approach (mid-range)
ALLOWED_TIERS = [PriceTier.LOW, PriceTier.MID, PriceTier.HIGH]

# Option 3: Conservative (high-value only)
ALLOWED_TIERS = [PriceTier.HIGH, PriceTier.MEGA]

# Option 4: Mix for diversification (recommended)
ALLOWED_TIERS = [PriceTier.LOW, PriceTier.MID, PriceTier.HIGH]  # Skip penny (too risky) and mega (low volatility)
```

**Usage Example**:

```python
# Get NASDAQ 100 tickers
all_tickers = get_nasdaq_100_tickers()  # ['AAPL', 'MSFT', 'NVDA', ...]

# Fetch current prices
prices = {
    'AAPL': 182.50,   # Mid tier
    'NVDA': 458.20,   # High tier
    'SIRI': 3.45,     # Penny tier
    'COST': 520.30,   # Mega tier
}

# Filter by allowed tiers
filtered_tickers = filter_by_price_tier(
    tickers=all_tickers,
    prices=prices,
    allowed_tiers=[PriceTier.MID, PriceTier.HIGH]
)
# Result: ['AAPL', 'NVDA'] (excludes SIRI and COST)

# Get characteristics for dynamic risk management
price = 182.50
tier = infer_price_tier(price)  # Returns PriceTier.MID
characteristics = get_price_tier_characteristics(tier)
# {
#     'avg_daily_volatility': 2.0,
#     'sentiment_weight': 1.2,
#     'min_volume': 2_000_000,
#     'position_size_limit': 0.10,
#     'stop_loss_pct': 0.03,
#     'ideal_horizon': 'swing_trade'
# }

# Use characteristics in trading logic
stop_loss_price = price * (1 - characteristics['stop_loss_pct'])  # $177.03
max_position_value = portfolio_value * characteristics['position_size_limit']  # 10% of portfolio
```

**Dashboard Integration**:

```python
# Show stocks grouped by price tier
for tier in ALLOWED_TIERS:
    tier_stocks = filter_by_price_tier(all_tickers, prices, [tier])
    print(f"\n{tier.value.upper()} TIER ({len(tier_stocks)} stocks)")
    for ticker in tier_stocks:
        print(f"  {ticker}: ${prices[ticker]:.2f}")
```

**Strategy Recommendations by Tier**:

```python
# Penny stocks: Quick day trades, high sentiment weight
if tier == PriceTier.PENNY:
    strategy = {
        'horizon': 'day_trade',
        'hold_time': '1-4 hours',
        'sentiment_weight': 2.0,  # News matters a LOT
        'position_size': '3-5% of portfolio',
        'stop_loss': '10%',
        'take_profit': '15-20%'
    }

# Mid-range: Swing trades, balanced
elif tier == PriceTier.MID:
    strategy = {
        'horizon': 'swing_trade',
        'hold_time': '3-7 days',
        'sentiment_weight': 1.2,
        'position_size': '8-10% of portfolio',
        'stop_loss': '3%',
        'take_profit': '5-10%'
    }

# Mega-cap: Long-term holds, stability
elif tier == PriceTier.MEGA:
    strategy = {
        'horizon': 'position',
        'hold_time': '2-4 weeks',
        'sentiment_weight': 0.8,  # Less reactive to news
        'position_size': '12-15% of portfolio',
        'stop_loss': '2%',
        'take_profit': '8-12%'
    }
```

**Automatic Risk Adjustment**:

The TFT model will learn different patterns for different price tiers through the `price_tier` categorical feature. This enables:

✅ **Tier-specific volatility expectations**
✅ **Dynamic stop-loss levels** (10% for penny, 2% for mega)
✅ **Adaptive sentiment weighting** (2.0x for penny, 0.8x for mega)
✅ **Position sizing based on risk** (5% for penny, 15% for mega)

---

### 1.5 Data Source Strategy

**Three Data Streams Required**:

#### Stream 1: Price & Volume Data
**Source**: Yahoo Finance (free, reliable)
```python
# Use yfinance library
import yfinance as yf

def fetch_price_data(ticker: str, period: str = '1y') -> pd.DataFrame:
    """
    Fetch historical price data.

    Args:
        ticker: Stock ticker (e.g., 'AAPL')
        period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y')

    Returns:
        DataFrame with columns: open, high, low, close, volume
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval='1h')  # Hourly data
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]
```

**Intervals Available**:
- `1m` - 1 minute (7 days max)
- `5m` - 5 minutes (60 days max)
- `15m` - 15 minutes (60 days max)
- `1h` - 1 hour (730 days max)
- `1d` - 1 day (unlimited)

**Strategy**:
- Historical training: Use `1h` interval for past 2 years
- Real-time inference: Use `5m` interval for intraday predictions

#### Stream 2: Technical Indicators
**Source**: Compute from price data using `ta-lib` or `pandas-ta`
```python
import pandas_ta as ta

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators from OHLCV data.

    Args:
        df: DataFrame with open, high, low, close, volume

    Returns:
        DataFrame with added technical indicators
    """
    # RSI (14-period)
    df['RSI'] = ta.rsi(df['close'], length=14)

    # MACD (12, 26, 9)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']

    # Bollinger Bands (20-period, 2 std)
    bbands = ta.bbands(df['close'], length=20, std=2)
    df['bollinger_upper'] = bbands['BBU_20_2.0']
    df['bollinger_lower'] = bbands['BBL_20_2.0']

    # Moving Averages
    df['SMA_20'] = ta.sma(df['close'], length=20)
    df['SMA_50'] = ta.sma(df['close'], length=50)

    # ATR (14-period volatility)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    return df
```

#### Stream 3: Sentiment Analysis (YOUR SECRET SAUCE)
**Source**: LM Studio with local models

**News Sources to Scrape**:
1. **Yahoo Finance News** (free API)
2. **Seeking Alpha** (RSS feeds)
3. **Reuters Business** (free articles)
4. **Twitter/X** (via API - $100/month)
5. **Reddit r/stocks, r/wallstreetbets** (free scraping)

**Sentiment Pipeline**:
```python
import requests
from datetime import datetime, timedelta

def fetch_recent_news(ticker: str, hours_back: int = 24) -> List[Dict]:
    """
    Fetch recent news articles for ticker.

    Args:
        ticker: Stock ticker
        hours_back: How many hours of history to fetch

    Returns:
        List of articles: [{'title': str, 'text': str, 'timestamp': datetime}, ...]
    """
    # Yahoo Finance News API (free)
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}"
    response = requests.get(url)
    news = response.json().get('news', [])

    articles = []
    cutoff = datetime.now() - timedelta(hours=hours_back)

    for article in news:
        pub_time = datetime.fromtimestamp(article['providerPublishTime'])
        if pub_time > cutoff:
            articles.append({
                'title': article['title'],
                'text': article.get('summary', ''),
                'timestamp': pub_time,
                'source': article.get('publisher', 'unknown')
            })

    return articles

def analyze_sentiment_lmstudio(text: str) -> float:
    """
    Analyze sentiment using LM Studio local model.

    Args:
        text: News article text

    Returns:
        Sentiment score 0-100 (0=very bearish, 50=neutral, 100=very bullish)
    """
    # LM Studio API endpoint (local)
    url = "http://localhost:1234/v1/chat/completions"

    prompt = f"""Analyze the sentiment of this financial news for stock trading.
Return ONLY a number from 0-100 where:
- 0-20 = Very Bearish (strong sell signal)
- 20-40 = Bearish (weak sell signal)
- 40-60 = Neutral (no clear signal)
- 60-80 = Bullish (weak buy signal)
- 80-100 = Very Bullish (strong buy signal)

News: {text}

Sentiment Score (0-100):"""

    payload = {
        "model": "local-model",  # Your LM Studio model
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,  # Low temp for consistent scoring
        "max_tokens": 10
    }

    response = requests.post(url, json=payload)
    result = response.json()['choices'][0]['message']['content'].strip()

    # Extract number from response
    import re
    score = re.search(r'\d+', result)
    return float(score.group()) if score else 50.0

def aggregate_sentiment(ticker: str, hours_back: int = 24) -> Dict:
    """
    Aggregate sentiment from multiple news sources.

    Returns:
        {
            'sentiment_score': 0-100,
            'sentiment_momentum': float (rate of change),
            'sentiment_volatility': float (std dev),
            'news_volume': int (number of articles)
        }
    """
    articles = fetch_recent_news(ticker, hours_back)

    if not articles:
        return {
            'sentiment_score': 50.0,  # Neutral if no news
            'sentiment_momentum': 0.0,
            'sentiment_volatility': 0.0,
            'news_volume': 0
        }

    # Analyze each article
    sentiments = []
    for article in articles:
        score = analyze_sentiment_lmstudio(article['title'] + ' ' + article['text'])
        sentiments.append({
            'score': score,
            'timestamp': article['timestamp']
        })

    # Calculate aggregate metrics
    scores = [s['score'] for s in sentiments]
    avg_sentiment = sum(scores) / len(scores)

    # Momentum: recent sentiment - older sentiment
    recent_scores = [s['score'] for s in sentiments[:len(sentiments)//2]]
    older_scores = [s['score'] for s in sentiments[len(sentiments)//2:]]
    momentum = (sum(recent_scores)/len(recent_scores) - sum(older_scores)/len(older_scores)) if older_scores else 0

    # Volatility: standard deviation
    import statistics
    volatility = statistics.stdev(scores) if len(scores) > 1 else 0

    return {
        'sentiment_score': avg_sentiment,
        'sentiment_momentum': momentum,
        'sentiment_volatility': volatility,
        'news_volume': len(articles)
    }
```

**LM Studio Model Recommendations**:
- **Mistral-7B-Instruct** - Fast, good reasoning
- **Llama-3-8B-Instruct** - Better financial understanding
- **Phi-3-Medium** - Very fast, decent accuracy

**Optimization**: Cache sentiment scores for 1 hour to avoid re-analyzing same articles

---

## Phase 2: Model Training & Validation (Week 3-4)

### 2.1 Training Data Generation

**Create**: `stock_data_fetcher.py` (replaces `metrics_generator.py`)

```python
# Fetch historical data for NASDAQ 100
def build_training_dataset(
    tickers: List[str],
    start_date: str = '2023-01-01',
    end_date: str = '2025-10-14',
    interval: str = '1h'
) -> pd.DataFrame:
    """
    Build comprehensive training dataset.

    Returns DataFrame with columns:
        - timestamp
        - ticker
        - sector
        - open, high, low, close, volume
        - RSI, MACD, MACD_signal, bollinger_upper, bollinger_lower, SMA_20, SMA_50, ATR
        - sentiment_score, sentiment_momentum, sentiment_volatility, news_volume
        - VIX, sector_performance, market_correlation
    """
    pass  # Implementation details
```

### 2.2 TFT Model Configuration

**Modify**: `tft_trainer.py`

Key changes:
```python
from stock_features import STOCK_FEATURES, TECHNICAL_FEATURES, SENTIMENT_FEATURES
from sector_profiles import SectorProfile, PriceTier

# Replace server metrics with stock features
time_varying_unknown_reals = STOCK_FEATURES.copy()

# Replace server profiles with sectors + add price tier
static_categoricals = [
    'ticker',      # Individual stock identity
    'sector',      # Tech, Finance, Healthcare, etc.
    'price_tier',  # Penny, Low, Mid, High, Mega (NEW!)
]

# Multi-target prediction (price, high, low)
target = ['close', 'high', 'low']

# Multi-horizon: predict 1hr, 4hr, 1day, 3day, 1week ahead
max_prediction_length = 168  # 1 week in hours
prediction_lengths = [1, 4, 24, 72, 168]  # 1hr, 4hr, 1day, 3day, 1week

# Training data will include price_tier column
# The model will learn that:
# - Penny stocks have high volatility, sensitive to sentiment
# - Mid stocks are balanced
# - Mega stocks are stable, less reactive to news
```

### 2.3 Backtesting Framework

**Create**: `backtest_engine.py`

```python
def backtest_strategy(
    predictions: pd.DataFrame,
    actual_prices: pd.DataFrame,
    strategy: str = 'hybrid'  # 'long_term', 'day_trade', 'hybrid'
) -> Dict:
    """
    Backtest trading strategy.

    Returns:
        {
            'total_return': float,
            'sharpe_ratio': float,
            'max_drawdown': float,
            'win_rate': float,
            'num_trades': int,
            'avg_hold_time': timedelta
        }
    """
    pass
```

---

## Phase 3: Trading Signal Generation (Week 5)

### 3.1 Buy/Sell Signal Logic

**Create**: `trading_signals.py`

```python
from enum import Enum
from typing import Dict, List

class SignalType(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

class TradeHorizon(Enum):
    DAY_TRADE = "day_trade"        # 1-2 days
    SWING_TRADE = "swing_trade"    # 3-7 days
    POSITION_TRADE = "position"    # 1-4 weeks

def generate_trading_signal(
    ticker: str,
    current_price: float,
    predictions: Dict,  # {p10, p50, p90} for multiple horizons
    sentiment: Dict,
    technical: Dict
) -> Dict:
    """
    Generate buy/sell signal with entry/exit points.

    Returns:
        {
            'signal': SignalType,
            'horizon': TradeHorizon,
            'confidence': 0-100,
            'entry_price': float,
            'target_price': float (take-profit),
            'stop_loss': float,
            'reasoning': str
        }
    """

    # === LONG-TERM HOLD CRITERIA ===
    # Strong fundamentals + bullish sentiment + upward trend
    if (
        sentiment['sentiment_score'] > 70 and
        sentiment['sentiment_momentum'] > 5 and
        technical['RSI'] < 65 and  # Not overbought
        predictions['1week']['p50'] > current_price * 1.05  # 5%+ predicted gain
    ):
        return {
            'signal': SignalType.STRONG_BUY,
            'horizon': TradeHorizon.POSITION_TRADE,
            'confidence': 85,
            'entry_price': current_price,
            'target_price': predictions['1week']['p50'],
            'stop_loss': predictions['1week']['p10'],
            'reasoning': 'Strong bullish sentiment + positive 1-week outlook'
        }

    # === DAY TRADE CRITERIA ===
    # Momentum play + high volatility + clear direction
    if (
        technical['RSI'] > 55 and technical['RSI'] < 70 and
        sentiment['sentiment_momentum'] > 10 and  # Rapid sentiment change
        predictions['4hr']['p50'] > current_price * 1.02  # 2%+ predicted gain
    ):
        return {
            'signal': SignalType.BUY,
            'horizon': TradeHorizon.DAY_TRADE,
            'confidence': 70,
            'entry_price': current_price,
            'target_price': predictions['4hr']['p50'],
            'stop_loss': current_price * 0.98,  # 2% stop-loss
            'reasoning': 'Momentum surge detected, 4hr upside predicted'
        }

    # === SELL CRITERIA ===
    # Overbought + negative sentiment shift
    if (
        technical['RSI'] > 75 and
        sentiment['sentiment_momentum'] < -10 and
        predictions['1day']['p50'] < current_price * 0.98
    ):
        return {
            'signal': SignalType.SELL,
            'horizon': TradeHorizon.DAY_TRADE,
            'confidence': 75,
            'entry_price': current_price,
            'target_price': predictions['1day']['p50'],
            'stop_loss': current_price * 1.02,
            'reasoning': 'Overbought + negative sentiment shift'
        }

    # Default: HOLD
    return {
        'signal': SignalType.HOLD,
        'horizon': TradeHorizon.SWING_TRADE,
        'confidence': 50,
        'entry_price': current_price,
        'target_price': current_price,
        'stop_loss': current_price * 0.95,
        'reasoning': 'No clear signal - wait for better setup'
    }
```

### 3.2 Portfolio Management

**Create**: `portfolio_manager.py`

```python
class Portfolio:
    """
    Manage portfolio of stocks with position sizing.
    """

    def __init__(self, initial_capital: float = 100000):
        self.capital = initial_capital
        self.positions = {}  # {ticker: {'shares': int, 'entry_price': float, 'stop_loss': float}}
        self.trade_history = []

    def calculate_position_size(
        self,
        ticker: str,
        entry_price: float,
        stop_loss: float,
        risk_per_trade: float = 0.02  # Risk 2% of capital per trade
    ) -> int:
        """
        Calculate number of shares to buy based on risk management.

        Kelly Criterion + Position Sizing
        """
        risk_amount = self.capital * risk_per_trade
        risk_per_share = entry_price - stop_loss
        shares = int(risk_amount / risk_per_share)

        # Cap position at 10% of portfolio
        max_position_value = self.capital * 0.10
        max_shares = int(max_position_value / entry_price)

        return min(shares, max_shares)

    def execute_trade(self, ticker: str, signal: Dict):
        """Execute buy/sell based on signal."""
        pass
```

---

## Phase 4: Real-Time Dashboard (Week 6)

### 4.1 Dashboard Layout

**Modify**: `tft_dashboard_web.py`

**Key Sections**:
1. **Active Positions** - Current holdings with P&L
2. **Buy/Sell Signals** - Top 10 opportunities (sorted by confidence)
3. **Watchlist** - NASDAQ 100 stocks with live predictions
4. **Sentiment Heatmap** - Visualize sentiment across sectors
5. **Portfolio Performance** - Cumulative returns, Sharpe ratio, drawdown
6. **Trade History** - Past trades with outcomes

**Example Layout**:
```
┌─────────────────────────────────────────────────────────────┐
│ NASDAQ TFT Trading Dashboard                    Portfolio: $125,340 (+25.3%)
├─────────────────────────────────────────────────────────────┤
│ 🟢 ACTIVE POSITIONS (5)                                     │
│ ┌──────┬────────┬────────┬──────────┬──────────┬──────────┐│
│ │Ticker│ Shares │  Entry │  Current │      P&L │   Action ││
│ ├──────┼────────┼────────┼──────────┼──────────┼──────────┤│
│ │ NVDA │    150 │ $420.5 │  $458.20 │  +$5,655 │ HOLD ✋  ││
│ │ AAPL │    200 │ $178.2 │  $182.50 │    +$860 │ HOLD ✋  ││
│ └──────┴────────┴────────┴──────────┴──────────┴──────────┘│
│                                                             │
│ 🎯 TOP BUY SIGNALS (Confidence > 70%)                      │
│ ┌──────┬─────────┬────────┬──────────┬──────────┬─────────┐│
│ │Ticker│  Signal │   Conf │    Entry │   Target │ Horizon ││
│ ├──────┼─────────┼────────┼──────────┼──────────┼─────────┤│
│ │ MSFT │ STR BUY │    85% │  $378.50 │  $395.20 │ 1-week  ││
│ │ GOOGL│     BUY │    78% │  $142.30 │  $148.60 │ 3-day   ││
│ │ META │     BUY │    72% │  $512.80 │  $524.10 │ 1-day   ││
│ └──────┴─────────┴────────┴──────────┴──────────┴─────────┘│
│                                                             │
│ 📊 SENTIMENT HEATMAP                                        │
│ Technology:        ████████████░░░░░░░░ 68/100 (Bullish)   │
│ Healthcare:        ███████░░░░░░░░░░░░░ 45/100 (Neutral)   │
│ Finance:           ████░░░░░░░░░░░░░░░░ 32/100 (Bearish)   │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 5: Production Deployment (Week 7-8)

### 5.1 Real-Time Data Pipeline

**Components**:
1. **Price Fetcher** - Updates every 5 minutes during market hours
2. **Sentiment Analyzer** - Scans news every 15 minutes
3. **TFT Inference Daemon** - Generates predictions every 30 minutes
4. **Signal Generator** - Updates buy/sell signals every 30 minutes
5. **Dashboard** - Real-time display

**Cron Jobs** (Windows Task Scheduler or Linux cron):
```bash
# Fetch price data (every 5 min during market hours 9:30am-4pm ET)
*/5 9-16 * * 1-5 python fetch_prices.py

# Analyze sentiment (every 15 min)
*/15 * * * * python analyze_sentiment.py

# Generate predictions (every 30 min)
*/30 * * * * python tft_inference_daemon.py

# Update signals (every 30 min)
*/30 * * * * python generate_signals.py
```

### 5.2 Risk Management & Alerts

**Create**: `risk_monitor.py`

```python
def check_risk_limits():
    """Monitor portfolio risk and send alerts."""

    # Check daily loss limit
    if daily_pnl < -5000:
        send_alert("CRITICAL: Daily loss limit reached (-$5,000)")

    # Check position concentration
    for ticker, position in portfolio.positions.items():
        position_pct = position['value'] / portfolio.total_value
        if position_pct > 0.15:
            send_alert(f"WARNING: {ticker} position exceeds 15% of portfolio")

    # Check stop-loss violations
    for ticker, position in portfolio.positions.items():
        current_price = get_current_price(ticker)
        if current_price < position['stop_loss']:
            send_alert(f"STOP-LOSS TRIGGERED: {ticker} @ ${current_price}")
            execute_sell(ticker)
```

---

## Migration Checklist

### Code Migration Steps

- [ ] **Week 1**: Create new project `StockTFT` from clean copy
- [ ] **Week 1**: Replace `linborg_schema.py` → `stock_features.py`
- [ ] **Week 1**: Replace `server_profiles.py` → `sector_profiles.py`
- [ ] **Week 1**: Replace `metrics_generator.py` → `stock_data_fetcher.py`
- [ ] **Week 2**: Set up data sources (yfinance, news APIs, LM Studio)
- [ ] **Week 2**: Build sentiment analysis pipeline
- [ ] **Week 3**: Adapt `tft_trainer.py` for stock features
- [ ] **Week 3**: Generate training data (2 years historical)
- [ ] **Week 3**: Train initial TFT model
- [ ] **Week 4**: Build backtesting framework
- [ ] **Week 4**: Validate model performance (Sharpe > 1.5 target)
- [ ] **Week 5**: Create `trading_signals.py` logic
- [ ] **Week 5**: Build portfolio manager
- [ ] **Week 6**: Adapt `tft_dashboard_web.py` for trading
- [ ] **Week 6**: Add sentiment heatmap visualizations
- [ ] **Week 7**: Set up real-time data pipeline
- [ ] **Week 7**: Deploy inference daemon
- [ ] **Week 8**: Paper trading with real-time data (no actual money)
- [ ] **Week 8**: Monitor performance for 2 weeks before live trading

### Infrastructure Requirements

**Hardware**:
- [ ] GPU recommended (NVIDIA RTX 3060+ or cloud GPU)
- [ ] 32GB RAM minimum
- [ ] 500GB SSD for historical data storage

**Software**:
- [ ] Python 3.10+
- [ ] PyTorch 2.0+
- [ ] LM Studio (for local sentiment analysis)
- [ ] Streamlit (dashboard)
- [ ] yfinance (price data)
- [ ] pandas-ta (technical indicators)

**Services**:
- [ ] LM Studio running locally (localhost:1234)
- [ ] Optional: Twitter API ($100/month for real-time tweets)
- [ ] Optional: Alpha Vantage (free tier for market data)

---

## Success Metrics

### Model Performance Targets

**Backtesting (Historical)**:
- Sharpe Ratio > 1.5 (good risk-adjusted returns)
- Win Rate > 55% (more winning trades than losing)
- Max Drawdown < 15% (limit catastrophic losses)
- Annual Return > 20% (beat S&P500)

**Sentiment Analysis**:
- Accuracy > 70% (sentiment direction matches price movement)
- Lead Time: Sentiment should predict price movement 4-24 hours ahead

**Live Trading (Paper)**:
- 2 weeks of paper trading before real money
- Must maintain Sharpe > 1.2 in live conditions
- No single-day loss > 3% of portfolio

---

## Risk Warnings

⚠️ **This is NOT financial advice**
⚠️ **Past performance does NOT guarantee future results**
⚠️ **Stock markets are MORE efficient than server monitoring** - edges disappear fast
⚠️ **Sentiment analysis is NOISY** - fake news, sarcasm, contradictions
⚠️ **Start with paper trading** - prove it works before risking real money
⚠️ **Use stop-losses ALWAYS** - protect against catastrophic losses
⚠️ **Diversify** - never put >10% in a single stock
⚠️ **Market regime changes** - 2020 COVID, 2022 rate hikes broke many models

---

## Next Steps

1. **Review this plan** - Discuss with your team
2. **Set up new repository** - `git clone` → `StockTFT`
3. **Install dependencies** - See infrastructure requirements
4. **Phase 1: Foundation** - Build `stock_features.py` and `sector_profiles.py`
5. **Phase 1: Sentiment** - Set up LM Studio and test sentiment analysis on 10 sample articles
6. **Phase 2: Training** - Fetch 2 years of NASDAQ 100 data and train initial model
7. **Phase 3: Backtesting** - Validate performance on 2024 data (out-of-sample)
8. **Phase 4: Dashboard** - Build monitoring interface
9. **Phase 5: Paper Trading** - 2 weeks of live testing with fake money
10. **Phase 6: Live Trading** - Start small ($10k) and scale if profitable

---

## Questions to Answer Before Starting

1. **Compute**: Do you have a GPU for training, or will you use cloud (AWS/GCP)?
2. **Sentiment Models**: Which LM Studio model do you want to use? (Mistral-7B, Llama-3-8B, Phi-3)
3. **News Sources**: Which news APIs will you pay for? (Twitter $100/month, or free only?)
4. **Trading Horizon**: Primary focus on day trading or long-term holds?
5. **Risk Tolerance**: How much capital are you willing to risk? ($10k? $50k? $100k?)
6. **Broker Integration**: Will you manually execute trades, or integrate with broker API (Alpaca, Interactive Brokers)?

---

## Conclusion

Your TFT server monitoring system is **95% ready** to adapt for stock trading. The architecture (centralized schema, profile detection, multi-horizon predictions, uncertainty bands) is PERFECT for this use case.

The hard parts:
1. **Sentiment analysis quality** - Garbage in, garbage out
2. **Market efficiency** - Harder to find edge than in server monitoring
3. **Regime changes** - Models need frequent retraining

The easy parts:
1. **Price data** - Free and clean (yfinance)
2. **Technical indicators** - Standard libraries (pandas-ta)
3. **TFT architecture** - Already built and tested
4. **Dashboard framework** - Already built

**Estimated timeline**: 8 weeks from zero to paper trading.

**Good luck, and may your Sharpe ratio be ever in your favor!** 📈🚀
