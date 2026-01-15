# Supertrend Multi-Strategy Trading System v5.0

Optimiertes Trading-System mit mehreren Strategien, automatischer Stock-Kategorisierung und IB Paper Trading.

## Features

- **Multi-Strategy**: SUPERTREND, BUY_HOLD, TREND_FOLLOW, GERMAN
- **Auto-Kategorisierung**: Backtest-basierte Zuweisung der optimalen Strategie
- **IB Paper Trading**: Automatischer Handel über Interactive Brokers
- **Trailing Stop Optimierung**: 20% für BUY_HOLD (backtested)

---

## Strategien

| Strategie | Beschreibung | Signal | Exit |
|-----------|--------------|--------|------|
| **SUPERTREND** | Volatile/Seitwärts-Aktien | Supertrend-Indikator | Signal + 12% Trailing |
| **BUY_HOLD** | Starke Bull-Runs | Immer Long | 20% Trailing + Re-Entry |
| **TREND_FOLLOW** | ETFs/Stabile Aktien | EMA 20/50 Cross | Signal + Stop |
| **GERMAN** | Deutsche Aktien (IBIS) | Supertrend | Signal + Stop |

---

## Aktuelle Portfolios (Top 10 je Kategorie)

**SUPERTREND** (schlägt B&H im Backtest):
```
PDYN, QBTS, MRNA, PYPL, PFE, NKE, MRK, TGT, UNH, NFLX
```

**BUY_HOLD** (starke Aufwärtstrends):
```
PLTR, NVDA, MSTR, QUBT, COIN, AVGO, MU, META, CRWD, SHOP
```

**TREND_FOLLOW** (ETFs):
```
SPY, QQQ, JPM, JNJ
```

**GERMAN** (IBIS/EUR):
```
TKMS
```

---

## Installation

```bash
# Virtual Environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Dependencies
pip install pandas numpy yfinance plotly ib_insync
```

---

## Usage

### 1. Signale prüfen (ohne IB)

```bash
python new5.py --screen
```

### 2. Stock-Kategorien verwalten

```bash
# Kategorien anzeigen
python categorize_stocks.py --list

# Ticker hinzufügen
python categorize_stocks.py --add AAPL GOOGL

# Ticker verschieben
python categorize_stocks.py --move TSLA BUY_HOLD

# Ticker entfernen
python categorize_stocks.py --remove SNAP

# Auto-Kategorisierung (Backtest-basiert)
python categorize_stocks.py --apply
```

### 3. Paper Trading (IB)

```bash
# Voraussetzung: TWS/Gateway läuft auf Port 7497

# Dry-Run (Test ohne Orders)
python ib_paper_trader.py --dry-run

# Live Paper Trading
python ib_paper_trader.py

# Nur Status
python ib_paper_trader.py --status

# Außerhalb Marktzeiten traden
python ib_paper_trader.py --force
```

---

## Konfiguration

Die Ticker und Einstellungen werden in `stock_categories.json` gespeichert:

```json
{
    "strategies": {
        "SUPERTREND": {
            "settings": {
                "st_period": 15,
                "st_multiplier": 4.0,
                "trailing_stop_pct": 0.12
            },
            "tickers": ["PDYN", "QBTS", ...]
        },
        "BUY_HOLD": {
            "settings": {
                "trailing_stop_pct": 0.20,
                "reentry_after_days": 5
            },
            "tickers": ["PLTR", "NVDA", ...]
        }
    }
}
```

---

## Backtest-Ergebnisse

### SUPERTREND vs Buy & Hold (3 Jahre)

| Ticker | Supertrend | B&H | Outperform |
|--------|------------|-----|------------|
| PDYN | +452% | +38% | **+414%** |
| QBTS | +2966% | +2878% | **+88%** |
| MRNA | 0% | -81% | **+81%** |
| PYPL | +21% | -22% | **+44%** |

### BUY_HOLD Trailing Stop Optimierung

| Stop % | Avg Return | vs B&H |
|--------|------------|--------|
| 15% | +877% | -164% |
| **20%** | **+1018%** | **-22%** |
| 25% | +949% | -91% |

→ **20% Trailing Stop** ist optimal (nur -2% hinter reinem B&H, aber mit Crash-Schutz)

---

## Marktzeiten

Das System handelt nur während NYSE/NASDAQ Öffnungszeiten:
- **15:30 - 22:00 Berlin Zeit**
- **09:30 - 16:00 US Eastern Time**

Mit `--force` Flag kann außerhalb der Zeiten gehandelt werden.

---

## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `new5.py` | Backtest-System |
| `ib_paper_trader.py` | IB Paper Trading |
| `categorize_stocks.py` | Ticker-Kategorisierung |
| `stock_categories.json` | Ticker & Settings |
| `ib_paper_trader_state.json` | Trading State |

---

## Strategie-Logik

### Wann SUPERTREND?
- Volatile Aktien mit häufigen Trendwechseln
- Seitwärts-/Bärenmärkte
- Aktien wo Supertrend B&H im Backtest schlägt

### Wann BUY_HOLD?
- Starke Aufwärtstrends (NVDA, PLTR, META)
- Parabolische Bull-Runs
- Bei diesen Aktien kann Supertrend B&H nicht schlagen

### Wann TREND_FOLLOW?
- ETFs (SPY, QQQ)
- Stabile Blue Chips
- EMA-Crossover als Signal

---

## Quick Start

```bash
# 1. Setup
python -m venv venv && source venv/bin/activate
pip install pandas numpy yfinance plotly ib_insync

# 2. Kategorien prüfen
python categorize_stocks.py --list

# 3. TWS starten (Paper Trading, Port 7497)

# 4. Paper Trading starten
python ib_paper_trader.py
```

---

## Lizenz

MIT License
