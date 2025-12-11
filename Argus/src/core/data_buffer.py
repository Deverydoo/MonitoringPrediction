#!/usr/bin/env python3
"""
Data Buffer - Accumulates inference metrics for retraining

Stores incoming metrics in daily parquet files for:
1. Automatic model retraining
2. Drift detection
3. Historical analysis
4. Incremental learning

Features:
- Automatic daily file rotation
- Configurable retention policy
- Efficient parquet storage
- Quick windowed access for training
"""

import pandas as pd
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import List, Dict, Optional
import logging
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataBuffer:
    """
    Accumulates incoming metrics for automated retraining.

    Stores metrics in daily parquet files with automatic rotation
    and retention management.
    """

    def __init__(
        self,
        buffer_dir: str = './data_buffer',
        retention_days: int = 60,
        auto_rotate: bool = True
    ):
        """
        Initialize data buffer.

        Args:
            buffer_dir: Directory to store parquet files
            retention_days: How many days of data to keep
            auto_rotate: Automatically rotate to new file at midnight
        """
        self.buffer_dir = Path(buffer_dir)
        self.buffer_dir.mkdir(parents=True, exist_ok=True)

        self.retention_days = retention_days
        self.auto_rotate = auto_rotate

        self.current_date = datetime.now().date()
        self.current_file = self._get_file_path(self.current_date)
        self.current_buffer: List[Dict] = []

        logger.info(f"📦 DataBuffer initialized (dir={buffer_dir}, retention={retention_days} days)")

    def _get_file_path(self, date_obj: date) -> Path:
        """Get parquet file path for a specific date."""
        filename = f"metrics_{date_obj.isoformat()}.parquet"
        return self.buffer_dir / filename

    def append(self, records: List[Dict]):
        """
        Append incoming metrics to buffer.

        Args:
            records: List of metric dictionaries
        """
        # Check if we need to rotate to new day
        if self.auto_rotate:
            self._rotate_if_needed()

        # Add to in-memory buffer
        self.current_buffer.extend(records)

        # Flush if buffer gets large (10,000 records ≈ 33 minutes at 5s intervals)
        if len(self.current_buffer) >= 10000:
            self.flush()

    def flush(self):
        """Write in-memory buffer to disk."""
        if not self.current_buffer:
            return

        # Convert to DataFrame
        df = pd.DataFrame(self.current_buffer)

        # Append to existing file or create new
        if self.current_file.exists():
            # Append mode
            existing_df = pd.read_parquet(self.current_file)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_parquet(self.current_file, index=False)
            logger.debug(f"📝 Appended {len(df)} records to {self.current_file.name}")
        else:
            # New file
            df.to_parquet(self.current_file, index=False)
            logger.info(f"📝 Created new buffer file: {self.current_file.name} ({len(df)} records)")

        # Clear buffer
        self.current_buffer.clear()

    def _rotate_if_needed(self):
        """Check if we need to rotate to a new day's file."""
        today = datetime.now().date()

        if today != self.current_date:
            logger.info(f"🔄 Day changed: {self.current_date} → {today}")

            # Flush current buffer before rotating
            self.flush()

            # Update to new date
            self.current_date = today
            self.current_file = self._get_file_path(today)

            # Clean up old files
            self._cleanup_old_files()

    def _cleanup_old_files(self):
        """Delete files older than retention period."""
        cutoff_date = datetime.now().date() - timedelta(days=self.retention_days)

        deleted_count = 0
        for file_path in self.buffer_dir.glob("metrics_*.parquet"):
            try:
                # Extract date from filename: metrics_2025-10-17.parquet
                date_str = file_path.stem.replace("metrics_", "")
                file_date = datetime.fromisoformat(date_str).date()

                if file_date < cutoff_date:
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"🗑️  Deleted old buffer file: {file_path.name}")

            except (ValueError, OSError) as e:
                logger.warning(f"⚠️  Error processing {file_path.name}: {e}")

        if deleted_count > 0:
            logger.info(f"✅ Cleaned up {deleted_count} old buffer files")

    def get_training_window(
        self,
        days: int = 30,
        include_today: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get combined DataFrame for training window.

        Args:
            days: Number of days to include
            include_today: Include today's incomplete data

        Returns:
            Combined DataFrame or None if no data
        """
        # Flush current buffer first
        self.flush()

        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days - 1)

        dfs = []

        # Collect all files in date range
        for i in range(days):
            date_to_check = start_date + timedelta(days=i)

            # Skip today if requested
            if not include_today and date_to_check == end_date:
                continue

            file_path = self._get_file_path(date_to_check)

            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    dfs.append(df)
                    logger.debug(f"📖 Loaded {len(df)} records from {file_path.name}")
                except Exception as e:
                    logger.warning(f"⚠️  Error reading {file_path.name}: {e}")

        if not dfs:
            logger.warning(f"⚠️  No data found for training window ({days} days)")
            return None

        # Combine all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)

        logger.info(f"✅ Training window ready: {len(combined_df)} records ({days} days)")

        return combined_df

    def get_stats(self) -> Dict:
        """
        Get buffer statistics.

        Returns:
            Dict with file count, total records, date range, disk usage
        """
        files = sorted(self.buffer_dir.glob("metrics_*.parquet"))

        if not files:
            return {
                'file_count': 0,
                'total_records': 0,
                'date_range': None,
                'disk_usage_mb': 0.0
            }

        # Extract dates
        dates = []
        total_records = 0
        total_size_bytes = 0

        for file_path in files:
            try:
                date_str = file_path.stem.replace("metrics_", "")
                file_date = datetime.fromisoformat(date_str).date()
                dates.append(file_date)

                df = pd.read_parquet(file_path)
                total_records += len(df)

                total_size_bytes += file_path.stat().st_size

            except Exception as e:
                logger.warning(f"⚠️  Error processing {file_path.name}: {e}")

        return {
            'file_count': len(files),
            'total_records': total_records,
            'date_range': {
                'start': min(dates).isoformat() if dates else None,
                'end': max(dates).isoformat() if dates else None
            },
            'disk_usage_mb': round(total_size_bytes / 1024 / 1024, 2),
            'buffer_dir': str(self.buffer_dir),
            'retention_days': self.retention_days
        }

    def export_training_data(
        self,
        output_path: str,
        days: int = 30
    ):
        """
        Export training window to single parquet file.

        Args:
            output_path: Where to save exported data
            days: How many days to include
        """
        df = self.get_training_window(days=days, include_today=False)

        if df is None:
            logger.error("❌ No data to export")
            return

        # Save to parquet
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(output_file, index=False)

        logger.info(f"✅ Exported {len(df)} records to {output_file}")

    def clear_all(self):
        """
        Clear all buffer files (USE WITH CAUTION).

        This deletes ALL accumulated data. Only use for testing.
        """
        files = list(self.buffer_dir.glob("metrics_*.parquet"))

        for file_path in files:
            file_path.unlink()

        self.current_buffer.clear()

        logger.warning(f"🗑️  Cleared all buffer files ({len(files)} files deleted)")


if __name__ == '__main__':
    # Example usage
    print("📦 DataBuffer - Example Usage\n")

    # Initialize buffer
    buffer = DataBuffer(buffer_dir='./test_data_buffer', retention_days=7)

    # Simulate adding metrics over multiple days
    print("Simulating 3 days of data...\n")

    for day in range(3):
        # Manually set date for testing
        test_date = datetime.now().date() - timedelta(days=2 - day)
        buffer.current_date = test_date
        buffer.current_file = buffer._get_file_path(test_date)

        print(f"Day {day + 1}: {test_date}")

        # Add 100 records per day
        records = []
        for i in range(100):
            records.append({
                'timestamp': datetime.now().isoformat(),
                'server_name': f'server{i % 10}',
                'cpu_pct': 50.0 + (i % 30),
                'mem_pct': 60.0 + (i % 20),
                'disk_io_mb_s': 100.0,
                'latency_ms': 50.0
            })

        buffer.append(records)
        buffer.flush()

        print(f"  Added {len(records)} records\n")

    # Get stats
    print("\nBuffer Statistics:")
    stats = buffer.get_stats()
    print(f"  Files: {stats['file_count']}")
    print(f"  Total Records: {stats['total_records']}")
    print(f"  Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    print(f"  Disk Usage: {stats['disk_usage_mb']} MB")

    # Get training window
    print("\nRetrieving 2-day training window...")
    df = buffer.get_training_window(days=2, include_today=False)
    if df is not None:
        print(f"  Retrieved {len(df)} records")
        print(f"  Columns: {list(df.columns)}")

    # Export
    print("\nExporting training data...")
    buffer.export_training_data('test_export.parquet', days=3)

    # Cleanup test files
    print("\nCleaning up test files...")
    shutil.rmtree('./test_data_buffer', ignore_errors=True)
    Path('test_export.parquet').unlink(missing_ok=True)
    print("✅ Test complete!")
