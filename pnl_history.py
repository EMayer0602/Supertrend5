"""
PnL History Tracker - Records PnL values over time for capital curve visualization.

Since TWS doesn't provide historical PnL data, we record it ourselves.
"""

import csv
import os
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PnLSnapshot:
    """A snapshot of PnL at a point in time."""
    timestamp: datetime
    net_liquidation: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    position_count: int


class PnLHistory:
    """Records and retrieves PnL history from CSV file."""

    def __init__(self, filepath: str = "pnl_history.csv"):
        self.filepath = filepath
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Create CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'net_liquidation', 'unrealized_pnl',
                    'realized_pnl', 'daily_pnl', 'position_count'
                ])

    def record(self, snapshot: PnLSnapshot):
        """Record a PnL snapshot to CSV."""
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                snapshot.timestamp.isoformat(),
                snapshot.net_liquidation,
                snapshot.unrealized_pnl,
                snapshot.realized_pnl,
                snapshot.daily_pnl,
                snapshot.position_count
            ])

    def record_from_monitor(self, monitor):
        """Record current PnL from a TradeMonitor instance."""
        snapshot = PnLSnapshot(
            timestamp=datetime.now(),
            net_liquidation=monitor.initial_capital,
            unrealized_pnl=monitor.tws_unrealized_pnl or 0,
            realized_pnl=monitor.tws_realized_pnl or 0,
            daily_pnl=monitor.tws_daily_pnl or 0,
            position_count=len(monitor.open_trades)
        )
        self.record(snapshot)
        return snapshot

    def get_history(self, hours: int = 24) -> List[PnLSnapshot]:
        """Get PnL history for the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        snapshots = []

        try:
            with open(self.filepath, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ts = datetime.fromisoformat(row['timestamp'])
                        if ts >= cutoff:
                            snapshots.append(PnLSnapshot(
                                timestamp=ts,
                                net_liquidation=float(row['net_liquidation']),
                                unrealized_pnl=float(row['unrealized_pnl']),
                                realized_pnl=float(row['realized_pnl']),
                                daily_pnl=float(row['daily_pnl']),
                                position_count=int(row['position_count'])
                            ))
                    except (ValueError, KeyError):
                        continue
        except FileNotFoundError:
            pass

        return snapshots

    def get_hourly_summary(self, hours: int = 24) -> List[Tuple[datetime, float]]:
        """Get hourly PnL summary (timestamp, total_pnl) for chart."""
        snapshots = self.get_history(hours)
        if not snapshots:
            return []

        # Group by hour and take last value of each hour
        hourly = {}
        for s in snapshots:
            hour_key = s.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly[hour_key] = s.unrealized_pnl + s.realized_pnl

        # Sort by timestamp
        return sorted(hourly.items())

    def get_daily_pnl_curve(self, hours: int = 8) -> List[Tuple[datetime, float]]:
        """Get daily PnL curve for intraday chart."""
        snapshots = self.get_history(hours)
        if not snapshots:
            return []

        return [(s.timestamp, s.daily_pnl) for s in snapshots]

    def get_total_pnl_curve(self, hours: int = 8) -> List[Tuple[datetime, float]]:
        """Get total PnL (realized + unrealized) curve for capital chart."""
        snapshots = self.get_history(hours)
        if not snapshots:
            return []

        return [(s.timestamp, s.unrealized_pnl + s.realized_pnl) for s in snapshots]

    def clear_old_data(self, days: int = 30):
        """Remove data older than N days."""
        cutoff = datetime.now() - timedelta(days=days)
        snapshots = []

        try:
            with open(self.filepath, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ts = datetime.fromisoformat(row['timestamp'])
                        if ts >= cutoff:
                            snapshots.append(row)
                    except (ValueError, KeyError):
                        continue
        except FileNotFoundError:
            return

        # Rewrite file with only recent data
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'net_liquidation', 'unrealized_pnl',
                'realized_pnl', 'daily_pnl', 'position_count'
            ])
            writer.writeheader()
            writer.writerows(snapshots)


# Convenience function
def record_pnl(monitor, filepath: str = "pnl_history.csv") -> PnLSnapshot:
    """Record current PnL from monitor to history file."""
    history = PnLHistory(filepath)
    return history.record_from_monitor(monitor)
