#!/usr/bin/env python3
"""
NordIQ AI Systems - Automated Retraining System
Nordic precision, AI intelligence

Copyright (c) 2025 NordIQ AI, LLC. All rights reserved.
Developed by Craig Giannelli

This software is licensed under the Business Source License 1.1.
See LICENSE file for details.

Automated Retraining System - Background Model Training

Handles automated model retraining triggered by:
1. Manual API call
2. Scheduled time (daily/weekly)
3. Data threshold (X days of new data)
4. Performance degradation detection

Features:
- Non-blocking background training (uses threading)
- Progress tracking and status monitoring
- Automatic model reload after training
- Training history and metadata tracking
- Configurable training schedules
"""

import os
import sys
import threading
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Callable
from enum import Enum

# Add training module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingStatus(Enum):
    """Training job status."""
    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingJob:
    """Represents a single training job with status tracking."""

    def __init__(
        self,
        job_id: str,
        dataset_path: str,
        epochs: int,
        incremental: bool = True
    ):
        self.job_id = job_id
        self.dataset_path = dataset_path
        self.epochs = epochs
        self.incremental = incremental

        self.status = TrainingStatus.QUEUED
        self.progress_pct = 0
        self.current_epoch = 0
        self.total_epochs = epochs
        self.error_message = None
        self.model_path = None

        self.started_at = None
        self.completed_at = None
        self.duration_seconds = None


class AutoRetrainer:
    """
    Automated retraining system with background job execution.

    Manages training jobs, tracks progress, and handles model reload.
    """

    def __init__(
        self,
        data_buffer,
        reload_callback: Optional[Callable[[str], Dict]] = None,
        training_days: int = 30,
        min_records_threshold: int = 100000
    ):
        """
        Initialize auto-retrainer.

        Args:
            data_buffer: DataBuffer instance for accessing training data
            reload_callback: Callback function to reload model after training
            training_days: How many days of data to use for training
            min_records_threshold: Minimum records needed to trigger retraining
        """
        self.data_buffer = data_buffer
        self.reload_callback = reload_callback
        self.training_days = training_days
        self.min_records_threshold = min_records_threshold

        # Job tracking
        self.current_job: Optional[TrainingJob] = None
        self.job_history: list[TrainingJob] = []
        self.training_lock = threading.Lock()

        # Statistics
        self.total_trainings = 0
        self.successful_trainings = 0
        self.failed_trainings = 0
        self.last_training_time: Optional[datetime] = None

        logger.info("🤖 AutoRetrainer initialized")

    def can_train(self) -> tuple[bool, str]:
        """
        Check if system is ready to train.

        Returns:
            (can_train, reason) tuple
        """
        # Check if training already in progress
        if self.current_job and self.current_job.status == TrainingStatus.RUNNING:
            return False, "Training already in progress"

        # Check if we have enough data
        stats = self.data_buffer.get_stats()
        if stats['total_records'] < self.min_records_threshold:
            return False, f"Insufficient data: {stats['total_records']} < {self.min_records_threshold} records"

        # Check if data buffer has required days
        if stats['file_count'] < self.training_days:
            return False, f"Not enough days: {stats['file_count']} < {self.training_days} days"

        return True, "Ready to train"

    def trigger_training(
        self,
        epochs: int = 5,
        incremental: bool = True,
        blocking: bool = False
    ) -> Dict:
        """
        Trigger a new training job.

        Args:
            epochs: Number of epochs to train
            incremental: Resume from latest checkpoint
            blocking: If True, wait for training to complete

        Returns:
            Status dict with job info
        """
        # Check if we can train
        can_train, reason = self.can_train()
        if not can_train:
            return {
                'success': False,
                'error': reason,
                'status': 'rejected'
            }

        # Acquire lock
        with self.training_lock:
            # Create job
            job_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            dataset_path = f"./training_buffer_{job_id}"

            job = TrainingJob(
                job_id=job_id,
                dataset_path=dataset_path,
                epochs=epochs,
                incremental=incremental
            )

            self.current_job = job

            logger.info(f"📋 Training job queued: {job_id}")

        # Start training in background thread
        if blocking:
            # Run synchronously
            self._execute_training(job)
            return self.get_job_status(job_id)
        else:
            # Run asynchronously
            thread = threading.Thread(
                target=self._execute_training,
                args=(job,),
                daemon=True
            )
            thread.start()

            return {
                'success': True,
                'job_id': job_id,
                'status': 'queued',
                'message': 'Training started in background',
                'epochs': epochs,
                'incremental': incremental
            }

    def _execute_training(self, job: TrainingJob):
        """
        Execute training job (runs in background thread).

        Args:
            job: TrainingJob to execute
        """
        try:
            logger.info(f"🚀 Starting training job: {job.job_id}")

            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.now()

            # Step 1: Export training data from buffer
            logger.info(f"[1/3] Exporting {self.training_days} days of data...")
            job.progress_pct = 10

            self.data_buffer.export_training_data(
                output_path=f"{job.dataset_path}/metrics.parquet",
                days=self.training_days
            )

            job.progress_pct = 30
            logger.info(f"✅ Training data exported to {job.dataset_path}")

            # Step 2: Train model
            logger.info(f"[2/3] Training model ({job.epochs} epochs)...")
            job.progress_pct = 40

            # Import training function
            from training.tft_trainer import train_model

            # Run training
            model_path = train_model(
                dataset_path=job.dataset_path,
                epochs=job.epochs,
                incremental=job.incremental
            )

            if not model_path:
                raise RuntimeError("Training failed - no model path returned")

            job.model_path = model_path
            job.progress_pct = 80
            logger.info(f"✅ Model trained: {model_path}")

            # Step 3: Reload model in daemon
            if self.reload_callback:
                logger.info(f"[3/3] Reloading model in daemon...")
                reload_result = self.reload_callback(model_path)

                if not reload_result.get('success'):
                    raise RuntimeError(f"Model reload failed: {reload_result.get('error')}")

                logger.info(f"✅ Model reloaded successfully")
            else:
                logger.warning("⚠️ No reload callback - model not automatically loaded")

            # Success!
            job.status = TrainingStatus.COMPLETED
            job.progress_pct = 100
            job.completed_at = datetime.now()
            job.duration_seconds = (job.completed_at - job.started_at).total_seconds()

            self.total_trainings += 1
            self.successful_trainings += 1
            self.last_training_time = job.completed_at

            logger.info(f"🎉 Training job completed: {job.job_id} ({job.duration_seconds:.0f}s)")

        except Exception as e:
            import traceback

            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            if job.started_at:
                job.duration_seconds = (job.completed_at - job.started_at).total_seconds()

            self.total_trainings += 1
            self.failed_trainings += 1

            logger.error(f"❌ Training job failed: {job.job_id}")
            logger.error(f"   Error: {e}")
            logger.error(traceback.format_exc())

        finally:
            # Move job to history
            self.job_history.append(job)
            if len(self.job_history) > 50:  # Keep last 50 jobs
                self.job_history.pop(0)

            # Clear current job
            if self.current_job and self.current_job.job_id == job.job_id:
                self.current_job = None

            # Cleanup temporary training data
            try:
                import shutil
                if Path(job.dataset_path).exists():
                    shutil.rmtree(job.dataset_path)
                    logger.info(f"🗑️ Cleaned up temp training data: {job.dataset_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup training data: {e}")

    def get_job_status(self, job_id: Optional[str] = None) -> Dict:
        """
        Get status of a training job.

        Args:
            job_id: Specific job ID, or None for current job

        Returns:
            Status dict
        """
        # If no job_id, return current job
        if job_id is None:
            if self.current_job is None:
                return {'status': 'idle', 'message': 'No active training job'}

            job = self.current_job
        else:
            # Find in history or current
            if self.current_job and self.current_job.job_id == job_id:
                job = self.current_job
            else:
                # Search history
                job = next((j for j in self.job_history if j.job_id == job_id), None)

                if job is None:
                    return {
                        'success': False,
                        'error': f'Job not found: {job_id}'
                    }

        return {
            'job_id': job.job_id,
            'status': job.status.value,
            'progress_pct': job.progress_pct,
            'epochs': job.epochs,
            'incremental': job.incremental,
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'duration_seconds': job.duration_seconds,
            'model_path': job.model_path,
            'error': job.error_message
        }

    def get_training_stats(self) -> Dict:
        """Get overall training statistics."""
        can_train, reason = self.can_train()

        stats = self.data_buffer.get_stats()

        return {
            'current_job': self.get_job_status() if self.current_job else None,
            'history': {
                'total_trainings': self.total_trainings,
                'successful': self.successful_trainings,
                'failed': self.failed_trainings,
                'last_training': self.last_training_time.isoformat() if self.last_training_time else None
            },
            'data_buffer': {
                'total_records': stats['total_records'],
                'file_count': stats['file_count'],
                'date_range': stats['date_range'],
                'disk_usage_mb': stats['disk_usage_mb']
            },
            'ready_to_train': can_train,
            'ready_reason': reason,
            'config': {
                'training_days': self.training_days,
                'min_records_threshold': self.min_records_threshold
            }
        }

    def cancel_current_job(self) -> Dict:
        """
        Cancel the currently running training job.

        Note: This is a soft cancel - training will continue but status will be marked cancelled.
        """
        if not self.current_job or self.current_job.status != TrainingStatus.RUNNING:
            return {
                'success': False,
                'error': 'No active training job to cancel'
            }

        self.current_job.status = TrainingStatus.CANCELLED
        logger.warning(f"⚠️ Training job cancelled: {self.current_job.job_id}")

        return {
            'success': True,
            'message': 'Training job marked as cancelled',
            'job_id': self.current_job.job_id
        }


if __name__ == '__main__':
    # Example usage
    print("🤖 AutoRetrainer - Example Usage\n")

    from core.data_buffer import DataBuffer

    # Initialize components
    data_buffer = DataBuffer(buffer_dir='./test_retraining_buffer', retention_days=30)
    retrainer = AutoRetrainer(
        data_buffer=data_buffer,
        reload_callback=lambda path: {'success': True, 'message': f'Mock reload: {path}'},
        training_days=7,
        min_records_threshold=1000
    )

    # Check if ready to train
    can_train, reason = retrainer.can_train()
    print(f"Ready to train: {can_train}")
    print(f"Reason: {reason}\n")

    # Get stats
    stats = retrainer.get_training_stats()
    print("Training Statistics:")
    print(f"  Ready: {stats['ready_to_train']}")
    print(f"  Reason: {stats['ready_reason']}")
    print(f"  Data Buffer: {stats['data_buffer']['total_records']} records")
    print(f"  Total Trainings: {stats['history']['total_trainings']}")

    # Cleanup
    import shutil
    shutil.rmtree('./test_retraining_buffer', ignore_errors=True)
    print("\n✅ Example complete!")
