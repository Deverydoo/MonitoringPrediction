#!/usr/bin/env python3
"""
Adaptive Retraining Daemon - Intelligent model retraining automation

Monitors drift metrics and automatically triggers retraining when:
1. Drift exceeds thresholds (88% SLA alignment)
2. Server load is in quiet period (to avoid impact)
3. Sufficient time has passed since last training (safeguards)

Implements safeguards:
- Minimum 6 hours between retraining attempts
- Maximum 30 days without retraining (force retrain)
- Only train during quiet periods (low server activity)
- Maximum 3 retraining attempts per week

This is the "brain" of the automated retraining pipeline.
"""

import argparse
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import logging
import json
import subprocess
import requests

from drift_monitor import DriftMonitor
from data_buffer import DataBuffer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RetrainingDecisionEngine:
    """
    Decides when to trigger model retraining based on multiple factors.

    Implements intelligent safeguards to prevent over-training and
    ensure retraining happens at optimal times.
    """

    def __init__(
        self,
        min_hours_between_training: int = 6,
        max_days_without_training: int = 30,
        max_trainings_per_week: int = 3,
        state_file: str = 'retraining_state.json'
    ):
        """
        Initialize decision engine.

        Args:
            min_hours_between_training: Minimum hours between retraining
            max_days_without_training: Force retrain after N days
            max_trainings_per_week: Maximum trainings per 7-day window
            state_file: Where to persist retraining state
        """
        self.min_hours_between_training = min_hours_between_training
        self.max_days_without_training = max_days_without_training
        self.max_trainings_per_week = max_trainings_per_week
        self.state_file = Path(state_file)

        # Load or initialize state
        self.state = self._load_state()

        logger.info(f"🧠 RetrainingDecisionEngine initialized")
        logger.info(f"   Min hours between training: {min_hours_between_training}")
        logger.info(f"   Max days without training: {max_days_without_training}")
        logger.info(f"   Max trainings per week: {max_trainings_per_week}")

    def _load_state(self) -> Dict:
        """Load retraining state from disk."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            logger.info(f"✅ Loaded retraining state from {self.state_file}")
            return state
        else:
            # Initialize new state
            state = {
                'last_training_time': None,
                'training_history': [],  # List of timestamps
                'total_trainings': 0,
                'last_drift_check': None
            }
            self._save_state(state)
            logger.info(f"📝 Created new retraining state at {self.state_file}")
            return state

    def _save_state(self, state: Optional[Dict] = None):
        """Save retraining state to disk."""
        if state is None:
            state = self.state

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

        logger.debug(f"💾 Saved retraining state")

    def should_retrain(
        self,
        drift_metrics: Dict,
        is_quiet_period: bool = True
    ) -> Dict:
        """
        Decide if retraining should be triggered.

        Args:
            drift_metrics: Current drift metrics from DriftMonitor
            is_quiet_period: Whether servers are in quiet period

        Returns:
            Dict with decision, reason, and safeguard status
        """
        now = datetime.now()
        decision = {
            'should_retrain': False,
            'reason': None,
            'safeguards': {},
            'timestamp': now.isoformat()
        }

        # Safeguard 1: Minimum time between trainings
        if self.state['last_training_time']:
            last_training = datetime.fromisoformat(self.state['last_training_time'])
            hours_since_last = (now - last_training).total_seconds() / 3600

            decision['safeguards']['hours_since_last_training'] = round(hours_since_last, 1)

            if hours_since_last < self.min_hours_between_training:
                decision['reason'] = f"Too soon (only {hours_since_last:.1f}h since last training, need {self.min_hours_between_training}h)"
                return decision

        # Safeguard 2: Maximum trainings per week
        week_ago = now - timedelta(days=7)
        recent_trainings = [
            t for t in self.state['training_history']
            if datetime.fromisoformat(t) > week_ago
        ]

        decision['safeguards']['trainings_this_week'] = len(recent_trainings)

        if len(recent_trainings) >= self.max_trainings_per_week:
            decision['reason'] = f"Weekly limit reached ({len(recent_trainings)}/{self.max_trainings_per_week} trainings this week)"
            return decision

        # Safeguard 3: Maximum days without training (FORCE RETRAIN)
        if self.state['last_training_time']:
            days_since_last = (now - last_training).total_seconds() / 86400

            decision['safeguards']['days_since_last_training'] = round(days_since_last, 1)

            if days_since_last > self.max_days_without_training:
                decision['should_retrain'] = True
                decision['reason'] = f"FORCE RETRAIN: {days_since_last:.1f} days since last training (max {self.max_days_without_training})"
                return decision

        # Check drift metrics
        needs_retraining_by_drift = drift_metrics.get('needs_retraining', False)
        combined_score = drift_metrics.get('combined_score', 0.0)

        decision['safeguards']['drift_score'] = round(combined_score, 3)
        decision['safeguards']['is_quiet_period'] = is_quiet_period

        if needs_retraining_by_drift:
            if not is_quiet_period:
                decision['reason'] = "Drift detected BUT not quiet period (waiting...)"
                return decision

            # All conditions met!
            decision['should_retrain'] = True
            decision['reason'] = f"Drift detected (score={combined_score:.2%}) and quiet period confirmed"
            return decision

        # No retraining needed
        decision['reason'] = f"Model healthy (drift score={combined_score:.2%})"
        return decision

    def record_training(self):
        """Record that a training was performed."""
        now = datetime.now().isoformat()

        self.state['last_training_time'] = now
        self.state['training_history'].append(now)
        self.state['total_trainings'] += 1

        # Keep only last 30 days of history
        cutoff = (datetime.now() - timedelta(days=30)).isoformat()
        self.state['training_history'] = [
            t for t in self.state['training_history']
            if t > cutoff
        ]

        self._save_state()

        logger.info(f"📝 Recorded training event (total: {self.state['total_trainings']})")


class AdaptiveRetrainingDaemon:
    """
    Main daemon that monitors drift and triggers retraining.

    Runs continuously in background, checking drift metrics
    and triggering retraining when conditions are met.
    """

    def __init__(
        self,
        daemon_url: str = 'http://localhost:8000',
        check_interval: int = 300,  # 5 minutes
        data_buffer_dir: str = './data_buffer',
        training_script: str = 'tft_trainer.py'
    ):
        """
        Initialize adaptive retraining daemon.

        Args:
            daemon_url: TFT inference daemon URL
            check_interval: Seconds between drift checks
            data_buffer_dir: Where data buffer stores metrics
            training_script: Path to training script
        """
        self.daemon_url = daemon_url
        self.check_interval = check_interval
        self.training_script = training_script

        # Initialize components
        self.drift_monitor = DriftMonitor(window_size=1000)
        self.data_buffer = DataBuffer(buffer_dir=data_buffer_dir)
        self.decision_engine = RetrainingDecisionEngine()

        self.running = False

        logger.info(f"🤖 AdaptiveRetrainingDaemon initialized")
        logger.info(f"   Daemon URL: {daemon_url}")
        logger.info(f"   Check interval: {check_interval}s ({check_interval/60:.1f} min)")

    def check_quiet_period(self) -> bool:
        """
        Check if servers are in quiet period (low activity).

        Simple heuristic: Check recent CPU/memory averages.
        Quiet period if average CPU < 60% and memory < 70%.

        Returns:
            True if quiet period, False otherwise
        """
        try:
            # Get recent predictions from daemon
            response = requests.get(
                f"{self.daemon_url}/predictions/current",
                timeout=5
            )

            if not response.ok:
                logger.warning(f"⚠️  Failed to get predictions: {response.status_code}")
                return False

            data = response.json()
            predictions = data.get('predictions', {})

            if not predictions:
                return False

            # Calculate average resource usage
            cpu_values = []
            mem_values = []

            for server_data in predictions.values():
                current = server_data.get('current', {})
                cpu_values.append(current.get('cpu_pct', 0))
                mem_values.append(current.get('mem_pct', 0))

            avg_cpu = sum(cpu_values) / len(cpu_values) if cpu_values else 0
            avg_mem = sum(mem_values) / len(mem_values) if mem_values else 0

            # Quiet period thresholds
            is_quiet = avg_cpu < 60.0 and avg_mem < 70.0

            if is_quiet:
                logger.info(f"✅ Quiet period detected (CPU={avg_cpu:.1f}%, MEM={avg_mem:.1f}%)")
            else:
                logger.debug(f"🔊 Active period (CPU={avg_cpu:.1f}%, MEM={avg_mem:.1f}%)")

            return is_quiet

        except Exception as e:
            logger.error(f"❌ Error checking quiet period: {e}")
            return False

    def trigger_retraining(self) -> bool:
        """
        Trigger incremental training with latest data.

        Returns:
            True if training succeeded, False otherwise
        """
        logger.info(f"🚀 Triggering incremental training...")

        try:
            # Export training data from buffer
            training_data_path = 'training_data_incremental.parquet'
            self.data_buffer.export_training_data(
                output_path=training_data_path,
                days=30  # Use last 30 days
            )

            # Run incremental training
            cmd = [
                sys.executable,
                self.training_script,
                '--data', training_data_path,
                '--incremental',
                '--epochs', '5'  # Add 5 more epochs
            ]

            logger.info(f"🔨 Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                logger.info(f"✅ Training completed successfully!")
                logger.info(f"📊 Training output:\n{result.stdout}")

                # Record training
                self.decision_engine.record_training()

                return True
            else:
                logger.error(f"❌ Training failed with exit code {result.returncode}")
                logger.error(f"Error output:\n{result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"❌ Training timed out (>1 hour)")
            return False
        except Exception as e:
            logger.error(f"❌ Error triggering training: {e}")
            return False

    def run_check_cycle(self):
        """Run one drift check and retraining decision cycle."""
        logger.info(f"🔍 Running drift check cycle...")

        # Calculate current drift metrics
        drift_metrics = self.drift_monitor.calculate_drift_metrics()

        logger.info(f"📊 Drift Metrics:")
        logger.info(f"   PER: {drift_metrics['per']:.2%}")
        logger.info(f"   DSS: {drift_metrics['dss']:.2%}")
        logger.info(f"   FDS: {drift_metrics['fds']:.2%}")
        logger.info(f"   Anomaly Rate: {drift_metrics['anomaly_rate']:.2%}")
        logger.info(f"   Combined Score: {drift_metrics['combined_score']:.2%}")

        # Check if quiet period
        is_quiet = self.check_quiet_period()

        # Decide if retraining needed
        decision = self.decision_engine.should_retrain(drift_metrics, is_quiet)

        logger.info(f"🧠 Decision: {decision['reason']}")

        if decision['should_retrain']:
            logger.info(f"🎯 Retraining triggered!")

            success = self.trigger_retraining()

            if success:
                logger.info(f"✅ Retraining completed successfully")
            else:
                logger.error(f"❌ Retraining failed")

        # Save drift metrics
        self.drift_monitor.save_metrics()

    def run(self):
        """Run daemon continuously."""
        self.running = True

        logger.info(f"🏃 Adaptive Retraining Daemon started")
        logger.info(f"   Check interval: {self.check_interval}s")
        logger.info(f"   Press Ctrl+C to stop")

        try:
            while self.running:
                self.run_check_cycle()

                logger.info(f"⏸️  Sleeping for {self.check_interval}s...")
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            logger.info(f"⚠️  Received interrupt signal, shutting down...")
            self.running = False
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}", exc_info=True)
            self.running = False

        logger.info(f"🛑 Adaptive Retraining Daemon stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Adaptive Retraining Daemon - Automated model retraining'
    )
    parser.add_argument(
        '--daemon-url',
        default='http://localhost:8000',
        help='TFT inference daemon URL'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=300,
        help='Check interval in seconds (default: 300 = 5 min)'
    )
    parser.add_argument(
        '--data-buffer-dir',
        default='./data_buffer',
        help='Data buffer directory'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run one check cycle then exit (for testing)'
    )

    args = parser.parse_args()

    # Create daemon
    daemon = AdaptiveRetrainingDaemon(
        daemon_url=args.daemon_url,
        check_interval=args.interval,
        data_buffer_dir=args.data_buffer_dir
    )

    if args.once:
        logger.info(f"🧪 Running single check cycle (--once mode)")
        daemon.run_check_cycle()
        logger.info(f"✅ Check cycle complete")
    else:
        daemon.run()


if __name__ == '__main__':
    main()
