#!/usr/bin/env python3
"""
metrics_generator.py - Enhanced Metrics Generator with Efficient Storage
Generate realistic server metrics and save in efficient formats (Parquet, HDF5)
Can be used as both importable module and command-line tool
"""

import os
import json
import random
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

# Optional efficient storage imports
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

try:
    import tables
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import data converter for efficient storage
try:
    from data_converter import DataFormatConverter
except ImportError:
    DataFormatConverter = None


@dataclass
class ServerProfile:
    """Defines a server's baseline characteristics and behavior patterns."""
    name: str
    baseline_cpu: float = 25.0
    baseline_memory: float = 45.0
    baseline_disk: float = 35.0
    baseline_load: float = 1.2
    baseline_network: float = 1000.0
    volatility: float = 0.15  # How much metrics vary from baseline
    problem_child: bool = False  # Server with recurring issues
    heavy_usage: bool = False  # Server with generally higher load
    
    def get_baseline_metrics(self) -> Dict[str, float]:
        """Get baseline metrics for this server."""
        multiplier = 1.5 if self.heavy_usage else 1.0
        problem_multiplier = 1.3 if self.problem_child else 1.0
        
        return {
            'cpu_percent': self.baseline_cpu * multiplier * problem_multiplier,
            'memory_percent': self.baseline_memory * multiplier * problem_multiplier,
            'disk_percent': self.baseline_disk,
            'load_average': self.baseline_load * multiplier * problem_multiplier,
            'network_bytes_sent': self.baseline_network * multiplier,
            'network_bytes_recv': self.baseline_network * multiplier,
            'disk_read_bytes': 500000 * multiplier,
            'disk_write_bytes': 300000 * multiplier,
            'java_heap_usage': 55.0 * multiplier * problem_multiplier,
            'java_gc_time': 2.0 * problem_multiplier
        }


@dataclass 
class TimeFrame:
    """Defines a specific operational time period with characteristic patterns."""
    name: str
    duration_minutes: int
    cpu_range: Tuple[float, float]
    memory_range: Tuple[float, float] 
    disk_range: Tuple[float, float]
    load_range: Tuple[float, float]
    network_multiplier: Tuple[float, float]
    java_heap_range: Tuple[float, float]
    java_gc_range: Tuple[float, float]
    anomaly_probability: float = 0.0
    description: str = ""


class MetricsDatasetGenerator:
    """Enhanced metrics generator compatible with existing TFT system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.servers = self._create_server_profiles()
        self.timeframes = self._define_timeframes()
        self.poll_interval_seconds = 300  # 5 minutes (same as TFT system expects)
        self.samples_per_minute = 60 // self.poll_interval_seconds
        
        logger.info(f"📊 Initialized generator for {len(self.servers)} servers")
        logger.info(f"⏱️  Poll interval: {self.poll_interval_seconds}s")
    
    def _create_server_profiles(self) -> List[ServerProfile]:
        """Create diverse server profiles based on existing naming patterns."""
        servers = []
        
        # Production servers (higher load)
        for i in range(18, 35):
            name = f"pprva00a{i:04d}"
            servers.append(ServerProfile(
                name=name,
                baseline_cpu=30.0,
                baseline_memory=55.0,
                baseline_disk=45.0,
                baseline_load=2.0,
                heavy_usage=True,
                problem_child=(random.random() < 0.1)  # 10% chance
            ))
        
        # Staging servers (moderate load)
        for i in range(18, 28):
            name = f"psrva00a{i:04d}"
            servers.append(ServerProfile(
                name=name,
                baseline_cpu=20.0,
                baseline_memory=40.0,
                baseline_disk=30.0,
                baseline_load=1.0,
                heavy_usage=False,
                problem_child=(random.random() < 0.05)  # 5% chance
            ))
        
        # Compute servers (variable load)
        for i in range(10, 25):
            name = f"cppr{i:02d}a{random.randint(1000, 9999):04d}"
            servers.append(ServerProfile(
                name=name,
                baseline_cpu=25.0,
                baseline_memory=50.0,
                baseline_disk=35.0,
                baseline_load=1.5,
                volatility=0.25,  # More variable
                heavy_usage=(random.random() < 0.3),  # 30% chance
                problem_child=(random.random() < 0.08)  # 8% chance
            ))
        
        # Service servers (steady load) - 10 servers
        for i in range(10, 20):
            name = f"csrva{i:02d}a{random.randint(1000, 9999):04d}"
            servers.append(ServerProfile(
                name=name,
                baseline_cpu=15.0,
                baseline_memory=35.0,
                baseline_disk=25.0,
                baseline_load=0.8,
                volatility=0.1,  # Low variability
                heavy_usage=False,
                problem_child=(random.random() < 0.03)  # 3% chance
            ))
        
        logger.info(f"🖥️  Created {len(servers)} server profiles:")
        logger.info(f"   Heavy usage servers: {sum(1 for s in servers if s.heavy_usage)}")
        logger.info(f"   Problem child servers: {sum(1 for s in servers if s.problem_child)}")
        
        return servers
    
    def _define_timeframes(self) -> Dict[str, TimeFrame]:
        """Define various operational timeframes with realistic characteristics."""
        return {
            'idle': TimeFrame(
                name='idle',
                duration_minutes=random.randint(120, 300),  # 2-5 hours
                cpu_range=(5.0, 20.0),
                memory_range=(20.0, 40.0),
                disk_range=(10.0, 30.0),
                load_range=(0.1, 0.8),
                network_multiplier=(0.1, 0.3),
                java_heap_range=(20.0, 40.0),
                java_gc_range=(0.5, 2.0),
                anomaly_probability=0.02,
                description="Low activity period (nights, weekends)"
            ),
            
            'healthy': TimeFrame(
                name='healthy',
                duration_minutes=random.randint(60, 180),  # 1-3 hours
                cpu_range=(15.0, 35.0),
                memory_range=(30.0, 60.0),
                disk_range=(20.0, 50.0),
                load_range=(0.5, 2.0),
                network_multiplier=(0.5, 1.2),
                java_heap_range=(40.0, 70.0),
                java_gc_range=(1.0, 4.0),
                anomaly_probability=0.05,
                description="Normal business operations"
            ),
            
            'morning_spike': TimeFrame(
                name='morning_spike',
                duration_minutes=random.randint(30, 90),  # 30min-1.5hrs
                cpu_range=(40.0, 70.0),
                memory_range=(50.0, 80.0),
                disk_range=(30.0, 60.0),
                load_range=(2.0, 5.0),
                network_multiplier=(1.5, 3.0),
                java_heap_range=(60.0, 85.0),
                java_gc_range=(5.0, 12.0),
                anomaly_probability=0.15,
                description="Morning login rush, batch jobs starting"
            ),
            
            'heavy_load': TimeFrame(
                name='heavy_load',
                duration_minutes=random.randint(45, 120),  # 45min-2hrs
                cpu_range=(60.0, 85.0),
                memory_range=(70.0, 90.0),
                disk_range=(50.0, 80.0),
                load_range=(4.0, 8.0),
                network_multiplier=(2.0, 4.0),
                java_heap_range=(75.0, 95.0),
                java_gc_range=(8.0, 20.0),
                anomaly_probability=0.25,
                description="Peak usage periods, heavy processing"
            ),
            
            'critical_issue': TimeFrame(
                name='critical_issue',
                duration_minutes=random.randint(5, 30),  # 5-30 minutes
                cpu_range=(85.0, 99.0),
                memory_range=(90.0, 99.0),
                disk_range=(70.0, 95.0),
                load_range=(8.0, 20.0),
                network_multiplier=(5.0, 10.0),
                java_heap_range=(95.0, 99.0),
                java_gc_range=(25.0, 50.0),
                anomaly_probability=0.95,
                description="System under severe stress, impending failure"
            ),
            
            'recovery': TimeFrame(
                name='recovery',
                duration_minutes=random.randint(10, 20),  # 10-20 minutes
                cpu_range=(20.0, 50.0),
                memory_range=(25.0, 55.0),
                disk_range=(15.0, 40.0),
                load_range=(0.5, 2.0),
                network_multiplier=(0.3, 0.8),
                java_heap_range=(25.0, 50.0),
                java_gc_range=(1.0, 5.0),
                anomaly_probability=0.05,
                description="Post-reboot recovery, services starting up"
            )
        }
    
    def generate_dataset(self, total_hours: int = 168, output_file: str = None) -> Dict[str, Any]:
        """
        Generate dataset in the format expected by the TFT training system.
        
        Args:
            total_hours: Hours of data to generate
            output_file: Output file path (will use training/metrics_dataset.json if None)
            
        Returns:
            Dict containing the generated dataset
        """
        logger.info(f"🚀 Generating {total_hours} hours of server metrics data")
        
        # Generate timeframe sequence
        timeframe_sequence = self._generate_timeframe_sequence(total_hours)
        logger.info(f"📅 Generated timeframe sequence: {len(timeframe_sequence)} periods")
        
        # Generate training samples
        training_samples = []
        current_time = datetime.now()
        
        for timeframe_name, duration_minutes in timeframe_sequence:
            timeframe = self.timeframes[timeframe_name]
            samples_in_period = duration_minutes // 5  # One sample every 5 minutes
            
            logger.info(f"🔄 Generating {samples_in_period} samples for {timeframe_name} ({duration_minutes} minutes)")
            
            for minute_offset in range(0, duration_minutes, 5):
                sample_time = current_time + timedelta(minutes=minute_offset)
                
                # Generate metrics for each server at this timestamp
                for server in self.servers:
                    sample = self._generate_sample(server, timeframe, sample_time)
                    training_samples.append(sample)
            
            current_time += timedelta(minutes=duration_minutes)
        
        # Create dataset in expected format
        dataset = {
            'training_samples': training_samples,
            'metadata': self._create_metadata(total_hours, len(training_samples), timeframe_sequence)
        }
        
        # Save to file
        if output_file is None:
            output_file = str(Path('./training/metrics_dataset.json'))
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving dataset: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Dataset generation completed: {len(training_samples)} samples")
        return dataset
    
    def _generate_timeframe_sequence(self, total_hours: int) -> List[Tuple[str, int]]:
        """Generate a realistic sequence of timeframes over the specified period."""
        sequence = []
        remaining_minutes = total_hours * 60
        current_time = 0
        
        while remaining_minutes > 30:
            # Choose next timeframe based on current time and probabilities
            timeframe_name = self._choose_next_timeframe(current_time, sequence)
            timeframe = self.timeframes[timeframe_name]
            
            # Determine duration (with some randomization)
            base_duration = timeframe.duration_minutes
            actual_duration = max(5, int(base_duration * random.uniform(0.7, 1.3)))
            actual_duration = min(actual_duration, remaining_minutes)
            
            sequence.append((timeframe_name, actual_duration))
            remaining_minutes -= actual_duration
            current_time += actual_duration
        
        # Fill remaining time with healthy state
        if remaining_minutes > 0:
            sequence.append(('healthy', remaining_minutes))
        
        return sequence
    
    def _choose_next_timeframe(self, current_time_minutes: int, sequence: List[Tuple[str, int]]) -> str:
        """Choose the next timeframe based on time of day and previous states."""
        hour_of_day = (current_time_minutes // 60) % 24
        last_state = sequence[-1][0] if sequence else 'healthy'
        
        # Time-based probabilities
        if 0 <= hour_of_day < 6:  # Night hours (0-6 AM)
            probabilities = {
                'idle': 0.6,
                'healthy': 0.2,
                'morning_spike': 0.05,
                'heavy_load': 0.05,
                'critical_issue': 0.05,
                'recovery': 0.05
            }
        elif 6 <= hour_of_day < 9:  # Morning hours (6-9 AM)
            probabilities = {
                'morning_spike': 0.4,
                'healthy': 0.3,
                'heavy_load': 0.15,
                'idle': 0.05,
                'critical_issue': 0.05,
                'recovery': 0.05
            }
        elif 9 <= hour_of_day < 17:  # Business hours (9 AM - 5 PM)
            probabilities = {
                'healthy': 0.4,
                'heavy_load': 0.25,
                'morning_spike': 0.15,
                'critical_issue': 0.1,
                'idle': 0.05,
                'recovery': 0.05
            }
        else:  # Evening hours
            probabilities = {
                'idle': 0.4,
                'healthy': 0.3,
                'heavy_load': 0.15,
                'critical_issue': 0.08,
                'morning_spike': 0.05,
                'recovery': 0.02
            }
        
        # State transition logic
        if last_state == 'critical_issue':
            return 'recovery'
        elif last_state == 'recovery':
            return random.choices(['healthy', 'idle'], weights=[0.7, 0.3])[0]
        
        # Normal state transitions based on probabilities
        states = list(probabilities.keys())
        weights = list(probabilities.values())
        return random.choices(states, weights=weights)[0]
    
    def _generate_sample(self, server: ServerProfile, timeframe: TimeFrame, timestamp: datetime) -> Dict[str, Any]:
        """Generate a single training sample."""
        baseline = server.get_baseline_metrics()
        
        # Generate metrics based on timeframe
        metrics = self._calculate_metrics(baseline, timeframe, server)
        
        # Determine status and severity
        is_anomaly = (random.random() < timeframe.anomaly_probability or 
                     metrics['cpu_percent'] > 90 or 
                     metrics['memory_percent'] > 95 or 
                     metrics['java_heap_usage'] > 95)
        
        status = 'anomaly' if is_anomaly else 'normal'
        
        # Determine severity
        if (metrics['cpu_percent'] > 95 or metrics['memory_percent'] > 98 or 
            metrics['java_heap_usage'] > 98):
            severity = 'critical'
        elif (metrics['cpu_percent'] > 85 or metrics['memory_percent'] > 90 or 
              metrics['java_heap_usage'] > 90):
            severity = 'high'
        elif (metrics['cpu_percent'] > 70 or metrics['memory_percent'] > 80 or 
              metrics['java_heap_usage'] > 80):
            severity = 'medium'
        else:
            severity = 'low'
        
        # Generate explanation
        explanation = self._generate_explanation(server, timeframe, metrics, status)
        
        return {
            'id': f"{server.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
            'timestamp': timestamp.isoformat(),
            'server_name': server.name,
            'status': status,
            'timeframe': timeframe.name,
            'metrics': metrics,
            'explanation': explanation,
            'severity': severity,
            'server_profile': {
                'heavy_usage': server.heavy_usage,
                'problem_child': server.problem_child,
                'volatility': server.volatility
            }
        }
    
    def _calculate_metrics(self, baseline: Dict[str, float], timeframe: TimeFrame, server: ServerProfile) -> Dict[str, float]:
        """Calculate metrics for timeframe."""
        metrics = {}
        
        # CPU with baseline adjustment
        cpu_base = baseline['cpu_percent']
        cpu_min, cpu_max = timeframe.cpu_range
        cpu_adjusted = cpu_base * random.uniform(cpu_min/cpu_base if cpu_base > 0 else cpu_min, 
                                               cpu_max/cpu_base if cpu_base > 0 else cpu_max)
        metrics['cpu_percent'] = max(0, min(100, cpu_adjusted * (1 + random.gauss(0, server.volatility))))
        
        # Memory with baseline adjustment
        mem_base = baseline['memory_percent']
        mem_min, mem_max = timeframe.memory_range
        mem_adjusted = mem_base * random.uniform(mem_min/mem_base if mem_base > 0 else mem_min, 
                                               mem_max/mem_base if mem_base > 0 else mem_max)
        metrics['memory_percent'] = max(0, min(100, mem_adjusted * (1 + random.gauss(0, server.volatility))))
        
        # Disk usage (slower changing)
        disk_base = baseline['disk_percent']
        disk_min, disk_max = timeframe.disk_range
        disk_adjusted = disk_base * random.uniform(disk_min/disk_base if disk_base > 0 else disk_min, 
                                                 disk_max/disk_base if disk_base > 0 else disk_max)
        metrics['disk_percent'] = max(0, min(100, disk_adjusted * (1 + random.gauss(0, server.volatility/2))))
        
        # Load average
        load_base = baseline['load_average']
        load_min, load_max = timeframe.load_range
        load_adjusted = load_base * random.uniform(load_min/load_base if load_base > 0 else load_min, 
                                                 load_max/load_base if load_base > 0 else load_max)
        metrics['load_average'] = max(0, load_adjusted * (1 + random.gauss(0, server.volatility)))
        
        # Network I/O
        net_mult_min, net_mult_max = timeframe.network_multiplier
        net_multiplier = random.uniform(net_mult_min, net_mult_max)
        metrics['network_bytes_sent'] = baseline['network_bytes_sent'] * net_multiplier * (1 + random.gauss(0, server.volatility))
        metrics['network_bytes_recv'] = baseline['network_bytes_recv'] * net_multiplier * (1 + random.gauss(0, server.volatility))
        
        # Disk I/O (correlated with CPU load)
        io_multiplier = (metrics['cpu_percent'] / 50.0) * net_multiplier
        metrics['disk_read_bytes'] = baseline['disk_read_bytes'] * io_multiplier * (1 + random.gauss(0, server.volatility))
        metrics['disk_write_bytes'] = baseline['disk_write_bytes'] * io_multiplier * (1 + random.gauss(0, server.volatility))
        
        # Java metrics
        java_heap_min, java_heap_max = timeframe.java_heap_range
        metrics['java_heap_usage'] = max(0, min(100, random.uniform(java_heap_min, java_heap_max) * 
                                               (1 + random.gauss(0, server.volatility))))
        
        java_gc_min, java_gc_max = timeframe.java_gc_range
        metrics['java_gc_time'] = max(0, random.uniform(java_gc_min, java_gc_max) * 
                                    (1 + random.gauss(0, server.volatility)))
        
        return metrics
    
    def _generate_explanation(self, server: ServerProfile, timeframe: TimeFrame, 
                            metrics: Dict[str, float], status: str) -> str:
        """Generate human-readable explanation for the metrics."""
        server_type = "problem child" if server.problem_child else "heavy usage" if server.heavy_usage else "standard"
        
        if status == 'anomaly':
            issues = []
            if metrics['cpu_percent'] > 90:
                issues.append(f"CPU usage critically high at {metrics['cpu_percent']:.1f}%")
            if metrics['memory_percent'] > 95:
                issues.append(f"memory usage at {metrics['memory_percent']:.1f}%")
            if metrics['java_heap_usage'] > 95:
                issues.append(f"Java heap at {metrics['java_heap_usage']:.1f}%")
            if metrics['load_average'] > 10:
                issues.append(f"load average extremely high at {metrics['load_average']:.1f}")
            
            if issues:
                return (f"ANOMALY detected on {server_type} server {server.name} during {timeframe.name}: "
                       f"{', '.join(issues)}. Immediate attention required.")
            else:
                return (f"ANOMALY detected on {server_type} server {server.name} during {timeframe.name}. "
                       f"System behavior outside normal parameters.")
        else:
            return (f"Server {server.name} ({server_type}) operating normally during {timeframe.name}. "
                   f"CPU: {metrics['cpu_percent']:.1f}%, Memory: {metrics['memory_percent']:.1f}%, "
                   f"Load: {metrics['load_average']:.1f}")
    
    def _create_metadata(self, total_hours: int, total_samples: int, timeframe_sequence: List[Tuple[str, int]]) -> Dict:
        """Create metadata for the dataset."""
        # Calculate timeframe distribution
        sequence_summary = {}
        for timeframe_name, duration in timeframe_sequence:
            sequence_summary[timeframe_name] = sequence_summary.get(timeframe_name, 0) + duration
        
        anomaly_count = int(total_samples * 0.15)  # Approximate
        
        return {
            'generated_at': datetime.now().isoformat(),
            'total_samples': total_samples,
            'anomaly_samples': anomaly_count,
            'normal_samples': total_samples - anomaly_count,
            'anomaly_ratio': anomaly_count / total_samples if total_samples > 0 else 0,
            'time_span_hours': total_hours,
            'servers_count': len(self.servers),
            'format_version': '2.0',
            'enhanced': True,
            'timeframe_sequence': timeframe_sequence,
            'timeframe_distribution': sequence_summary,
            'poll_interval_seconds': self.poll_interval_seconds
        }


# Module interface functions
def generate_metrics_dataset(hours: int = 168, 
                           output_file: str = None,
                           config: Dict[str, Any] = None) -> Optional[str]:
    """
    Generate metrics dataset - designed for module usage.
    
    Args:
        hours: Hours of data to generate
        output_file: Optional output file path
        config: Optional configuration dictionary
        
    Returns:
        Path to generated file or None if failed
    """
    if config is None:
        config = {}
    
    generator = MetricsDatasetGenerator(config)
    result = generator.generate_dataset(hours, output_file)
    return output_file if result else None


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Enhanced metrics dataset generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 1 week of data
  python metrics_generator.py --hours 168
  
  # Generate large dataset with custom output
  python metrics_generator.py --hours 720 --output large_dataset.json
  
  # Generate with custom seed for reproducibility
  python metrics_generator.py --hours 168 --seed 42
        """
    )
    
    parser.add_argument(
        '--hours', '-t', type=int, default=168,
        help='Total hours of metrics to generate (default: 168 = 1 week)'
    )
    
    parser.add_argument(
        '--output', '-o', type=str,
        help='Output file path (default: ./training/metrics_dataset.json)'
    )
    
    parser.add_argument(
        '--seed', type=int,
        help='Random seed for reproducible generation'
    )
    
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set random seed
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        logger.info(f"🎲 Random seed set to {args.seed}")
    
    # Validate arguments
    if args.hours <= 0:
        logger.error("❌ Hours must be positive")
        return 1
    
    if args.hours > 8760:  # More than 1 year
        logger.warning(f"⚠️  Generating {args.hours} hours ({args.hours/24:.1f} days) - this will be a large dataset")
        try:
            confirm = input("Continue? (y/N): ")
            if confirm.lower() != 'y':
                return 0
        except KeyboardInterrupt:
            return 0
    
    try:
        # Generate dataset
        logger.info(f"🚀 Starting generation of {args.hours} hours of metrics data")
        
        start_time = time.time()
        
        result = generate_metrics_dataset(
            hours=args.hours,
            output_file=args.output,
            config={}
        )
        
        generation_time = time.time() - start_time
        
        if result:
            result_path = Path(result)
            file_size_mb = result_path.stat().st_size / (1024 * 1024)
            
            logger.info(f"\n🎉 GENERATION COMPLETED!")
            logger.info("=" * 50)
            logger.info(f"Time taken: {generation_time:.1f} seconds")
            logger.info(f"Output file: {result}")
            logger.info(f"File size: {file_size_mb:.1f} MB")
            logger.info(f"Time span: {args.hours} hours ({args.hours/24:.1f} days)")
            
            return 0
        else:
            logger.error("❌ Generation failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n⏹️  Generation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())