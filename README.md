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

## Dateien

- `new5.py` - Hauptskript mit allen Funktionen
- `supertrend_*.html` - Generierte Charts
- `venv/` - Virtual Environment (nicht committen)

---

## Lizenz

MIT License
