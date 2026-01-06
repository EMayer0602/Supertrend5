# Supertrend Trading System v5.0

Optimiertes Trading-System basierend auf dem Supertrend-Indikator mit HTF-Filter, RSI-Filterung und automatischem Stock-Screening.

## Installation

```bash
# Virtual Environment erstellen
python -m venv venv

# Aktivieren (Linux/Mac)
source venv/bin/activate

# Aktivieren (Windows)
venv\Scripts\activate

# Dependencies installieren
pip install pandas numpy yfinance plotly
```

## Usage Scenarios

### 1. Standard-Analyse (Einzelne Aktie)

Analysiert eine einzelne Aktie (Standard: MSFT) mit automatischer Parameter-Optimierung.

```bash
python new5.py
```

**Was passiert:**
- Lädt 5 Jahre Kursdaten
- Testet 8 verschiedene Strategie-Varianten
- Optimiert Supertrend-Parameter (Period, Multiplier)
- Vergleicht mit Buy & Hold
- Erstellt interaktiven HTML-Chart

**Output:**
- `supertrend_MSFT_results.html` - Interaktiver Chart mit allen Trades

---

### 2. Multi-Ticker-Analyse (Dow Jones 30 + NASDAQ Top 20)

Testet die Strategie auf 50 der wichtigsten US-Aktien.

```bash
python new5.py --multi
```

**Was passiert:**
- Testet alle 30 Dow Jones Komponenten
- Testet Top 20 NASDAQ-Aktien
- Zeigt Win-Rate gegen Buy & Hold
- Identifiziert beste Performer

**Typische Ergebnisse:**
- ~39% der Aktien schlagen Buy & Hold
- Beste bei volatilen Aktien (NFLX, META, TSLA)

---

### 3. Enhanced Analysis (Markt-Klassifizierung)

Erweiterte Analyse mit Volatilitäts- und Trend-Klassifizierung.

```bash
python new5.py --enhanced
```

**Was passiert:**
- Klassifiziert Aktien nach Volatilität (HIGH/MEDIUM/LOW)
- Analysiert Trend-Stärke (STRONG/MODERATE/WEAK)
- Gibt Empfehlung: SUPERTREND vs BUY_AND_HOLD
- Berechnet Recommendation Accuracy

**Erkenntnisse:**
| Volatilität | Win-Rate | Empfehlung |
|-------------|----------|------------|
| HIGH        | 75%      | Supertrend |
| MEDIUM      | 38%      | Gemischt   |
| LOW         | 0%       | Buy & Hold |

---

### 4. Stock Screener (80 Aktien)

Screent 80 populäre Aktien und findet die besten Kandidaten für Supertrend.

```bash
python new5.py --screen
```

**Was passiert:**
- Screent Tech, Finance, Healthcare, Consumer, Industrial, Energy Aktien
- Klassifiziert nach Eignung: EXCELLENT, GOOD, NEUTRAL, BUY_HOLD
- Gibt Top 5 Picks aus

**Typische Top-Picks:**
1. NFLX - +218% Outperformance
2. COIN - +204% Outperformance
3. SHOP - +156% Outperformance
4. META - +143% Outperformance
5. DKNG - +118% Outperformance

---

## Strategie-Varianten

Das System testet automatisch folgende Strategien:

| Strategie | HTF Filter | RSI | Trailing Stop |
|-----------|------------|-----|---------------|
| Long Only (Basic) | Nein | Nein | Nein |
| Long + Trailing 8% | Nein | Nein | 8% |
| Long + Trailing 12% | Nein | Nein | 12% |
| Long + RSI Filter | Nein | Ja | Nein |
| Long + RSI + Trail 12% | Nein | Ja | 12% |
| HTF Long Only | Ja | Nein | Nein |
| HTF + Trail 15% | Ja | Nein | 15% |

---

## Kernkonzepte

### Supertrend-Indikator
- Trend-Following-Indikator basierend auf ATR
- Generiert klare Buy/Sell-Signale
- Parameter: Period (5-35), Multiplier (1.5-8.0)

### HTF Filter (Higher Time Frame)
- Berechnet Supertrend auf Wochen-Basis
- Filtert Trades gegen den übergeordneten Trend
- Reduziert Fehlsignale in Seitwärtsmärkten

### RSI Filter
- Relative Strength Index (14 Perioden)
- Kauft nur wenn RSI < 70 (nicht überkauft)
- Vermeidet Einstiege an Hochpunkten

### Trailing Stop
- Dynamischer Stop-Loss der dem Kurs folgt
- Sichert Gewinne bei Trendfortsetzung
- Typisch: 8-15% unter Höchstkurs

---

## Wann Supertrend verwenden?

**Ideal für:**
- Volatile Aktien (NFLX, META, TSLA, COIN)
- Aktien mit häufigen Trendwechseln
- Bären-/Seitwärtsmärkte
- Drawdown-Schutz

**Nicht ideal für:**
- Stabile Aufwärtstrends (SPY, QQQ, JPM)
- Niedrig-volatile Blue Chips
- Extreme Bull-Runs (NVDA 2023-2024)

---

## Beispiel-Output

```
================================================================================
STRATEGY COMPARISON RESULTS (sorted by return)
================================================================================
Rank  Strategy                       Return    vs B&H   Trades
-----------------------------------------------------------------
1     Long + RSI Filter              122.0%    -5.3%       12
2     HTF Long Only                  118.5%    -8.8%        8
3     Long Only (Basic)              115.2%   -12.1%       15
...

>>> 2 strategies beat Buy & Hold!
```

---

## 5. IB Paper Trading (Live Trading Simulation)

Handelt die Top 30 Supertrend-Aktien automatisch im Interactive Brokers Paper Trading Account.

```bash
# Voraussetzung: ib_insync installieren
pip install ib_insync

# Dry-Run (Simulation ohne IB-Verbindung)
python ib_paper_trader.py --dry-run

# Echtes Paper Trading (TWS/Gateway muss laufen)
python ib_paper_trader.py

# Nur Status anzeigen
python ib_paper_trader.py --status
```

**Voraussetzungen:**
- Interactive Brokers Paper Trading Account
- TWS (Trader Workstation) oder IB Gateway läuft
- API-Verbindung aktiviert (Port 7497 für TWS Paper)

**Features:**
- Handelt automatisch die Top 30 volatilen Aktien
- 5% max pro Position, max 20 Positionen
- 8% Stop-Loss + 12% Trailing Stop
- Signale alle 5 Minuten aktualisiert
- Persistenter State (überlebt Neustart)

**Top 30 Aktien im Portfolio:**
```
NFLX, COIN, SHOP, META, DKNG, ARKK, MRNA, ROKU, PYPL, SNOW,
TSLA, AMD, RBLX, ADBE, BA, DIS, CRM, NKE, TGT, PFE,
UNH, LOW, SBUX, SQ, UBER, SNAP, PINS, DOCU, ZM, CRWD
```

---

## Dateien

- `new5.py` - Hauptskript mit allen Backtest-Funktionen
- `ib_paper_trader.py` - IB Paper Trading System
- `supertrend_*.html` - Generierte Charts
- `ib_paper_trader_state.json` - Trading State (Positionen, Signale)
- `ib_paper_trader.log` - Trading Log
- `venv/` - Virtual Environment (nicht committen)

---

## Quick Start

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate  # oder venv\Scripts\activate (Windows)
pip install pandas numpy yfinance plotly ib_insync

# 2. Screener laufen lassen um beste Aktien zu finden
python new5.py --screen

# 3. Paper Trading starten (erst --dry-run testen!)
python ib_paper_trader.py --dry-run
```

---

## Lizenz

MIT License
