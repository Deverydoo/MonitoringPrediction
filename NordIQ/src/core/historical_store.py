#!/usr/bin/env python3
"""
Historical Data Store - Parquet-Based Event and Metrics Storage
================================================================

Stores historical events for reporting and analysis:
- Alert events (critical, warning, resolved)
- Server state changes
- Environment health snapshots
- Incident tracking

Data is stored in Parquet for consistency with the rest of the system.
Designed for executive reporting with CSV export support.
"""

import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoricalDataStore:
    """
    Persistent storage for historical events and metrics using Parquet.

    Stores:
    - Alert events (when servers go critical/warning/resolved)
    - Environment snapshots (periodic health summaries)
    - Incident records (user-confirmed outages)

    Thread-safe for use with async daemons.
    """

    def __init__(self, data_dir: str = "./data"):
        """Initialize the historical data store."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.alerts_file = self.data_dir / "historical_alerts.parquet"
        self.snapshots_file = self.data_dir / "historical_snapshots.parquet"
        self.incidents_file = self.data_dir / "historical_incidents.parquet"

        self._lock = threading.Lock()

        # Initialize DataFrames (load existing or create empty)
        self._init_dataframes()
        logger.info(f"📊 HistoricalDataStore initialized at {self.data_dir}")

    def _init_dataframes(self):
        """Load existing data or create empty DataFrames."""
        # Alert events schema
        self.alerts_schema = {
            'id': 'int64',
            'timestamp': 'datetime64[ns]',
            'server_name': 'str',
            'event_type': 'str',
            'previous_level': 'str',
            'new_level': 'str',
            'risk_score': 'float64',
            'metrics_snapshot': 'str',  # JSON string
            'resolved_at': 'datetime64[ns]',
            'resolution_duration_minutes': 'float64',
            'caused_incident': 'bool',
            'notes': 'str'
        }

        # Environment snapshots schema
        self.snapshots_schema = {
            'id': 'int64',
            'timestamp': 'datetime64[ns]',
            'total_servers': 'int64',
            'critical_count': 'int64',
            'warning_count': 'int64',
            'degraded_count': 'int64',
            'healthy_count': 'int64',
            'prob_30m': 'float64',
            'prob_8h': 'float64',
            'avg_risk_score': 'float64',
            'max_risk_score': 'float64',
            'top_risk_server': 'str',
            'fleet_health': 'str'
        }

        # Load or create alerts DataFrame
        if self.alerts_file.exists():
            try:
                self.alerts_df = pd.read_parquet(self.alerts_file)
            except Exception as e:
                logger.warning(f"Could not load alerts file: {e}")
                self.alerts_df = self._create_empty_alerts_df()
        else:
            self.alerts_df = self._create_empty_alerts_df()

        # Load or create snapshots DataFrame
        if self.snapshots_file.exists():
            try:
                self.snapshots_df = pd.read_parquet(self.snapshots_file)
            except Exception as e:
                logger.warning(f"Could not load snapshots file: {e}")
                self.snapshots_df = self._create_empty_snapshots_df()
        else:
            self.snapshots_df = self._create_empty_snapshots_df()

    def _create_empty_alerts_df(self) -> pd.DataFrame:
        """Create empty alerts DataFrame with correct schema."""
        return pd.DataFrame({
            'id': pd.Series(dtype='int64'),
            'timestamp': pd.Series(dtype='datetime64[ns]'),
            'server_name': pd.Series(dtype='str'),
            'event_type': pd.Series(dtype='str'),
            'previous_level': pd.Series(dtype='str'),
            'new_level': pd.Series(dtype='str'),
            'risk_score': pd.Series(dtype='float64'),
            'metrics_snapshot': pd.Series(dtype='str'),
            'resolved_at': pd.Series(dtype='datetime64[ns]'),
            'resolution_duration_minutes': pd.Series(dtype='float64'),
            'caused_incident': pd.Series(dtype='bool'),
            'notes': pd.Series(dtype='str')
        })

    def _create_empty_snapshots_df(self) -> pd.DataFrame:
        """Create empty snapshots DataFrame with correct schema."""
        return pd.DataFrame({
            'id': pd.Series(dtype='int64'),
            'timestamp': pd.Series(dtype='datetime64[ns]'),
            'total_servers': pd.Series(dtype='int64'),
            'critical_count': pd.Series(dtype='int64'),
            'warning_count': pd.Series(dtype='int64'),
            'degraded_count': pd.Series(dtype='int64'),
            'healthy_count': pd.Series(dtype='int64'),
            'prob_30m': pd.Series(dtype='float64'),
            'prob_8h': pd.Series(dtype='float64'),
            'avg_risk_score': pd.Series(dtype='float64'),
            'max_risk_score': pd.Series(dtype='float64'),
            'top_risk_server': pd.Series(dtype='str'),
            'fleet_health': pd.Series(dtype='str')
        })

    def _save_alerts(self):
        """Save alerts DataFrame to Parquet."""
        try:
            self.alerts_df.to_parquet(self.alerts_file, index=False)
        except Exception as e:
            logger.error(f"Failed to save alerts: {e}")

    def _save_snapshots(self):
        """Save snapshots DataFrame to Parquet."""
        try:
            self.snapshots_df.to_parquet(self.snapshots_file, index=False)
        except Exception as e:
            logger.error(f"Failed to save snapshots: {e}")

    def record_alert_event(self, server_name: str, event_type: str,
                          previous_level: Optional[str], new_level: str,
                          risk_score: float, metrics: Optional[Dict] = None) -> int:
        """
        Record an alert state change event.

        Args:
            server_name: Name of the server
            event_type: 'escalation', 'de-escalation', 'resolved', 'new_alert'
            previous_level: Previous alert level (critical/warning/degraded/healthy)
            new_level: New alert level
            risk_score: Current risk score
            metrics: Optional dict of current metrics

        Returns:
            Event ID
        """
        with self._lock:
            # Generate new ID
            event_id = len(self.alerts_df) + 1

            # Create new row
            new_row = pd.DataFrame([{
                'id': event_id,
                'timestamp': pd.Timestamp.now(),
                'server_name': server_name,
                'event_type': event_type,
                'previous_level': previous_level or '',
                'new_level': new_level,
                'risk_score': risk_score,
                'metrics_snapshot': json.dumps(metrics) if metrics else '',
                'resolved_at': pd.NaT,
                'resolution_duration_minutes': None,
                'caused_incident': False,
                'notes': ''
            }])

            self.alerts_df = pd.concat([self.alerts_df, new_row], ignore_index=True)
            self._save_alerts()

            logger.info(f"📝 Alert event: {server_name} {event_type} -> {new_level} (score: {risk_score:.0f})")
            return event_id

    def resolve_alert(self, server_name: str, notes: Optional[str] = None,
                     caused_incident: bool = False):
        """Mark the most recent alert for a server as resolved."""
        with self._lock:
            # Find the most recent unresolved alert for this server
            mask = (
                (self.alerts_df['server_name'] == server_name) &
                (self.alerts_df['resolved_at'].isna())
            )

            if mask.any():
                idx = self.alerts_df[mask].index[-1]  # Most recent
                started_at = self.alerts_df.loc[idx, 'timestamp']
                now = pd.Timestamp.now()
                duration = (now - started_at).total_seconds() / 60

                self.alerts_df.loc[idx, 'resolved_at'] = now
                self.alerts_df.loc[idx, 'resolution_duration_minutes'] = duration
                self.alerts_df.loc[idx, 'caused_incident'] = caused_incident
                if notes:
                    self.alerts_df.loc[idx, 'notes'] = notes

                self._save_alerts()
                logger.info(f"✅ Alert resolved: {server_name} after {duration:.1f} minutes")

    def record_environment_snapshot(self, summary: Dict):
        """
        Record a periodic environment health snapshot.

        Args:
            summary: Dict with keys like total_servers, critical_count, etc.
        """
        with self._lock:
            # Generate new ID
            snapshot_id = len(self.snapshots_df) + 1

            # Create new row
            new_row = pd.DataFrame([{
                'id': snapshot_id,
                'timestamp': pd.Timestamp.now(),
                'total_servers': summary.get('total_servers', 0),
                'critical_count': summary.get('critical_count', 0),
                'warning_count': summary.get('warning_count', 0),
                'degraded_count': summary.get('degraded_count', 0),
                'healthy_count': summary.get('healthy_count', 0),
                'prob_30m': summary.get('prob_30m', 0),
                'prob_8h': summary.get('prob_8h', 0),
                'avg_risk_score': summary.get('avg_risk_score', 0),
                'max_risk_score': summary.get('max_risk_score', 0),
                'top_risk_server': summary.get('top_risk_server', ''),
                'fleet_health': summary.get('fleet_health', 'unknown')
            }])

            self.snapshots_df = pd.concat([self.snapshots_df, new_row], ignore_index=True)
            self._save_snapshots()

    def get_alert_events(self, time_range: str = "1h",
                        server_name: Optional[str] = None) -> List[Dict]:
        """
        Get alert events within a time range.

        Args:
            time_range: '30m', '1h', '8h', '1d', '1w', '1M'
            server_name: Optional filter by server

        Returns:
            List of alert event dicts
        """
        cutoff = self._get_cutoff_time(time_range)

        with self._lock:
            mask = self.alerts_df['timestamp'] >= cutoff

            if server_name:
                mask &= self.alerts_df['server_name'] == server_name

            filtered = self.alerts_df[mask].sort_values('timestamp', ascending=False)

            # Convert to list of dicts with proper serialization
            result = []
            for _, row in filtered.iterrows():
                record = row.to_dict()
                # Convert timestamps to ISO strings
                record['timestamp'] = row['timestamp'].isoformat() if pd.notna(row['timestamp']) else None
                record['resolved_at'] = row['resolved_at'].isoformat() if pd.notna(row['resolved_at']) else None
                result.append(record)

            return result

    def get_environment_snapshots(self, time_range: str = "1h") -> List[Dict]:
        """Get environment snapshots within a time range."""
        cutoff = self._get_cutoff_time(time_range)

        with self._lock:
            mask = self.snapshots_df['timestamp'] >= cutoff
            filtered = self.snapshots_df[mask].sort_values('timestamp', ascending=True)

            # Convert to list of dicts
            result = []
            for _, row in filtered.iterrows():
                record = row.to_dict()
                record['timestamp'] = row['timestamp'].isoformat() if pd.notna(row['timestamp']) else None
                result.append(record)

            return result

    def get_summary_stats(self, time_range: str = "1d") -> Dict:
        """
        Get summary statistics for executive reporting.

        Returns dict with:
        - total_alerts: Total alert events in period
        - critical_alerts: Number of critical alerts
        - resolved_count: How many were resolved
        - avg_resolution_time: Average time to resolution
        - incidents_count: Confirmed incidents
        - servers_affected: Unique servers with alerts
        """
        cutoff = self._get_cutoff_time(time_range)

        with self._lock:
            mask = self.alerts_df['timestamp'] >= cutoff
            filtered = self.alerts_df[mask]

            total_alerts = len(filtered)
            critical_alerts = len(filtered[filtered['new_level'] == 'critical'])
            warning_alerts = len(filtered[filtered['new_level'] == 'warning'])
            resolved = filtered['resolved_at'].notna().sum()
            unresolved = total_alerts - resolved
            incidents_caused = filtered['caused_incident'].sum() if 'caused_incident' in filtered.columns else 0
            servers_affected = filtered['server_name'].nunique()

            # Average resolution time
            resolved_rows = filtered[filtered['resolved_at'].notna()]
            avg_resolution_time = resolved_rows['resolution_duration_minutes'].mean() if len(resolved_rows) > 0 else 0

            resolution_rate = (resolved / total_alerts * 100) if total_alerts > 0 else 100

            return {
                'time_range': time_range,
                'total_alerts': int(total_alerts),
                'critical_alerts': int(critical_alerts),
                'warning_alerts': int(warning_alerts),
                'resolved_count': int(resolved),
                'unresolved_count': int(unresolved),
                'avg_resolution_minutes': round(avg_resolution_time, 1) if pd.notna(avg_resolution_time) else 0,
                'incidents_caused': int(incidents_caused),
                'servers_affected': int(servers_affected),
                'resolution_rate': round(resolution_rate, 1)
            }

    def get_server_history(self, server_name: str, time_range: str = "1d") -> Dict:
        """Get detailed history for a specific server."""
        cutoff = self._get_cutoff_time(time_range)

        with self._lock:
            mask = (
                (self.alerts_df['server_name'] == server_name) &
                (self.alerts_df['timestamp'] >= cutoff)
            )
            filtered = self.alerts_df[mask].sort_values('timestamp', ascending=False)

            # Convert to list of dicts
            events = []
            for _, row in filtered.iterrows():
                record = row.to_dict()
                record['timestamp'] = row['timestamp'].isoformat() if pd.notna(row['timestamp']) else None
                record['resolved_at'] = row['resolved_at'].isoformat() if pd.notna(row['resolved_at']) else None
                events.append(record)

            # Calculate stats
            stats = {
                'total_events': len(filtered),
                'critical_count': len(filtered[filtered['new_level'] == 'critical']),
                'resolved_count': filtered['resolved_at'].notna().sum(),
                'avg_resolution': filtered['resolution_duration_minutes'].mean() if len(filtered) > 0 else 0
            }

            return {
                'server_name': server_name,
                'time_range': time_range,
                'events': events,
                'stats': stats
            }

    def export_to_csv(self, table: str, time_range: str = "1d") -> str:
        """
        Export data to CSV format.

        Args:
            table: 'alerts', 'environment', 'summary'
            time_range: Time range to export

        Returns:
            CSV string
        """
        cutoff = self._get_cutoff_time(time_range)

        with self._lock:
            if table == 'alerts':
                mask = self.alerts_df['timestamp'] >= cutoff
                df = self.alerts_df[mask].sort_values('timestamp', ascending=False)
                # Select columns for export
                export_cols = ['timestamp', 'server_name', 'event_type', 'previous_level',
                              'new_level', 'risk_score', 'resolved_at',
                              'resolution_duration_minutes', 'caused_incident', 'notes']
                df = df[[c for c in export_cols if c in df.columns]]

            elif table == 'environment':
                mask = self.snapshots_df['timestamp'] >= cutoff
                df = self.snapshots_df[mask].sort_values('timestamp', ascending=False)
                # Select columns for export
                export_cols = ['timestamp', 'total_servers', 'critical_count', 'warning_count',
                              'degraded_count', 'healthy_count', 'prob_30m', 'prob_8h',
                              'avg_risk_score', 'max_risk_score', 'top_risk_server', 'fleet_health']
                df = df[[c for c in export_cols if c in df.columns]]

            else:
                return ""

            if df.empty:
                return ""

            return df.to_csv(index=False)

    def _get_cutoff_time(self, time_range: str) -> pd.Timestamp:
        """Convert time range string to cutoff timestamp."""
        now = pd.Timestamp.now()

        if time_range == '30m':
            return now - pd.Timedelta(minutes=30)
        elif time_range == '1h':
            return now - pd.Timedelta(hours=1)
        elif time_range == '8h':
            return now - pd.Timedelta(hours=8)
        elif time_range == '1d':
            return now - pd.Timedelta(days=1)
        elif time_range == '1w':
            return now - pd.Timedelta(weeks=1)
        elif time_range == '1M':
            return now - pd.Timedelta(days=30)
        else:
            return now - pd.Timedelta(hours=1)

    def cleanup_old_data(self, retention_days: int = 90):
        """Remove data older than retention period."""
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=retention_days)

        with self._lock:
            alerts_before = len(self.alerts_df)
            self.alerts_df = self.alerts_df[self.alerts_df['timestamp'] >= cutoff]
            alerts_deleted = alerts_before - len(self.alerts_df)

            snapshots_before = len(self.snapshots_df)
            self.snapshots_df = self.snapshots_df[self.snapshots_df['timestamp'] >= cutoff]
            snapshots_deleted = snapshots_before - len(self.snapshots_df)

            if alerts_deleted > 0:
                self._save_alerts()
            if snapshots_deleted > 0:
                self._save_snapshots()

            logger.info(f"🧹 Cleanup: Removed {alerts_deleted} alerts, {snapshots_deleted} snapshots older than {retention_days} days")


# Singleton instance for the daemon
_store_instance: Optional[HistoricalDataStore] = None


def get_historical_store(data_dir: str = "./data") -> HistoricalDataStore:
    """Get or create the singleton historical store instance."""
    global _store_instance
    if _store_instance is None:
        _store_instance = HistoricalDataStore(data_dir)
    return _store_instance


if __name__ == "__main__":
    # Test the store
    store = HistoricalDataStore("./test_data")

    # Record some test events
    store.record_alert_event(
        server_name="ppdb001",
        event_type="escalation",
        previous_level="warning",
        new_level="critical",
        risk_score=85.5,
        metrics={'cpu': 92, 'memory': 88}
    )

    store.record_environment_snapshot({
        'total_servers': 27,
        'critical_count': 2,
        'warning_count': 5,
        'degraded_count': 3,
        'healthy_count': 17,
        'prob_30m': 0.45,
        'prob_8h': 0.62,
        'avg_risk_score': 38.5,
        'max_risk_score': 85.5,
        'top_risk_server': 'ppdb001',
        'fleet_health': 'warning'
    })

    # Get summary
    summary = store.get_summary_stats('1h')
    print("\n📊 Summary Stats:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Export CSV
    csv_data = store.export_to_csv('alerts', '1h')
    print("\n📄 CSV Export Preview:")
    print(csv_data[:500] if csv_data else "No data")

    print("\n✅ Historical store test complete!")
    print(f"Files created in: ./test_data/")
