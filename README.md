# Backtesting_Data

Institutional Quantitative Backtesting Datasets for Multi-Asset Algorithmic Trading (Forex, CFDs, and Binance USDT-M Perpetuals).

---

## 📁 Repository Structure

### 1. `Forex_Data/`
- **Universe**: 156 Multi-Asset Instruments (Major FX pairs, Exotic crosses, Global Equity Indices, Commodities, and Industrial Metals).
- **Timeframes**: Multi-timeframe synchronized parquet files (`15m`, `1h`, `4h`, `d1`) spanning 2020–2026.
- **Coverage**: 626 parquet files with true UTC timestamps, complete OHLCV bars, and zero lookahead alignments.

### 2. `Binance_Data/`
- **Universe**: 18 Institutional Binance USDT-M Perpetuals:
  - `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT`, `DOGEUSDT`, `ADAUSDT`, `TRXUSDT`, `LINKUSDT`, `AVAXUSDT`, `SUIUSDT`, `NEARUSDT`, `DOTUSDT`, `LTCUSDT`, `BCHUSDT`, `APTUSDT`, `OPUSDT`, `ARBUSDT`
- **Datasets**:
  - `*_15m_master_2020_2026.parquet`: Master 15m OHLCV bars with funding rates and volume metrics.
  - `*_15m_footprint_ladder.parquet`: 100% verified tick footprint orderbook ladders.
  - `*_dataset_manifest.json`: Metadata manifests and verification checksums.

### 3. `s4_fvg_ml_strategy.py`
- Self-contained production strategy runner implementing the 4H trend + Daily sweep + 15m Fair Value Gap architecture across 20 Out-Of-Sample walk-forward regimes.
