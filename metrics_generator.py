#!/usr/bin/env python3
"""
Enhanced Metrics Dataset Generator
Standalone script for generating realistic time-series server metrics
Simulates various operational states with proper temporal patterns
"""

import os
import json
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import argparse
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    """Enhanced metrics dataset generator with realistic temporal patterns."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.servers = self._create_server_profiles()
        self.timeframes = self._define_timeframes()
        self.poll_interval_seconds = 5  # Poll every 5 seconds
        self.samples_per_minute = 60 // self.poll_interval_seconds  # 12 samples per minute
        
        # Dataset storage
        self.training_samples = []
        self.generation_start_time = datetime.now()
        
        logger.info(f"📊 Initialized generator for {len(self.servers)} servers")
        logger.info(f"⏱️  Poll interval: {self.poll_interval_seconds}s ({self.samples_per_minute} samples/minute)")
    
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
        
        # Service servers (steady load)
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
        
        # Container runtime servers
        for i in range(10, 15):
            name = f"crva{i:02d}a{random.randint(1000, 9999):04d}"
            servers.append(ServerProfile(
                name=name,
                baseline_cpu=35.0,
                baseline_memory=60.0,
                baseline_disk=40.0,
                baseline_load=2.5,
                volatility=0.2,
                heavy_usage=True,
                problem_child=(random.random() < 0.12)  # 12% chance
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
            
            'maintenance': TimeFrame(
                name='maintenance',
                duration_minutes=random.randint(15, 60),  # 15min-1hr
                cpu_range=(30.0, 60.0),
                memory_range=(40.0, 70.0),
                disk_range=(60.0, 90.0),  # High disk usage for updates
                load_range=(1.0, 4.0),
                network_multiplier=(0.5, 2.0),
                java_heap_range=(30.0, 60.0),
                java_gc_range=(2.0, 8.0),
                anomaly_probability=0.1,
                description="Maintenance operations, updates, backups"
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
            ),
            
            'offline': TimeFrame(
                name='offline',
                duration_minutes=random.randint(3, 10),  # 3-10 minutes
                cpu_range=(0.0, 0.0),
                memory_range=(0.0, 0.0),
                disk_range=(0.0, 0.0),
                load_range=(0.0, 0.0),
                network_multiplier=(0.0, 0.0),
                java_heap_range=(0.0, 0.0),
                java_gc_range=(0.0, 0.0),
                anomaly_probability=1.0,  # Offline is always an anomaly
                description="Server offline (reboot, crash, maintenance)"
            )
        }
    
    def _generate_timeframe_sequence(self, total_hours: int) -> List[Tuple[str, int]]:
        """Generate a realistic sequence of timeframes over the specified period."""
        sequence = []
        remaining_minutes = total_hours * 60
        
        # Start with a healthy state
        current_time = 0
        
        while remaining_minutes > 30:  # Ensure we have enough time for meaningful timeframes
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
            
            logger.debug(f"Added {timeframe_name} for {actual_duration} minutes")
        
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
                'idle': 0.5,
                'healthy': 0.2,
                'maintenance': 0.15,
                'morning_spike': 0.05,
                'heavy_load': 0.05,
                'critical_issue': 0.03,
                'recovery': 0.02
            }
        elif 6 <= hour_of_day < 9:  # Morning hours (6-9 AM)
            probabilities = {
                'morning_spike': 0.3,
                'healthy': 0.25,
                'heavy_load': 0.2,
                'idle': 0.1,
                'critical_issue': 0.08,
                'maintenance': 0.05,
                'recovery': 0.02
            }
        elif 9 <= hour_of_day < 17:  # Business hours (9 AM - 5 PM)
            probabilities = {
                'healthy': 0.35,
                'heavy_load': 0.25,
                'morning_spike': 0.15,
                'critical_issue': 0.1,
                'idle': 0.08,
                'maintenance': 0.05,
                'recovery': 0.02
            }
        elif 17 <= hour_of_day < 20:  # Evening hours (5-8 PM)
            probabilities = {
                'healthy': 0.3,
                'heavy_load': 0.2,
                'idle': 0.2,
                'maintenance': 0.15,
                'critical_issue': 0.08,
                'morning_spike': 0.05,
                'recovery': 0.02
            }
        else:  # Late evening (8 PM - midnight)
            probabilities = {
                'idle': 0.35,
                'healthy': 0.25,
                'maintenance': 0.2,
                'heavy_load': 0.1,
                'critical_issue': 0.05,
                'morning_spike': 0.03,
                'recovery': 0.02
            }
        
        # State transition logic
        if last_state == 'critical_issue':
            # Critical issues are often followed by offline or recovery
            return random.choices(['offline', 'recovery'], weights=[0.7, 0.3])[0]
        elif last_state == 'offline':
            # Offline is always followed by recovery
            return 'recovery'
        elif last_state == 'recovery':
            # Recovery typically leads to healthy or idle
            return random.choices(['healthy', 'idle'], weights=[0.7, 0.3])[0]
        
        # Normal state transitions based on probabilities
        states = list(probabilities.keys())
        weights = list(probabilities.values())
        return random.choices(states, weights=weights)[0]
    
    def _generate_metrics_for_timeframe(self, server: ServerProfile, timeframe: TimeFrame, 
                                      timestamp: datetime) -> Dict[str, Any]:
        """Generate realistic metrics for a server during a specific timeframe."""
        baseline = server.get_baseline_metrics()
        
        # Special handling for offline state
        if timeframe.name == 'offline':
            return {
                'id': f"{server.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
                'timestamp': timestamp.isoformat(),
                'server_name': server.name,
                'status': 'anomaly',
                'timeframe': timeframe.name,
                'metrics': {
                    'cpu_percent': 0.0,
                    'memory_percent': 0.0,
                    'disk_percent': 0.0,
                    'load_average': 0.0,
                    'network_bytes_sent': 0.0,
                    'network_bytes_recv': 0.0,
                    'disk_read_bytes': 0.0,
                    'disk_write_bytes': 0.0,
                    'java_heap_usage': 0.0,
                    'java_gc_time': 0.0,
                    'availability': 0.0  # Server is offline
                },
                'explanation': f"Server {server.name} is offline. No metrics available.",
                'severity': 'critical',
                'alert_type': 'server_offline'
            }
        
        # Generate metrics within timeframe ranges
        metrics = {}
        
        # CPU with baseline adjustment
        cpu_base = baseline['cpu_percent']
        cpu_min, cpu_max = timeframe.cpu_range
        cpu_adjusted = cpu_base * random.uniform(cpu_min/cpu_base, cpu_max/cpu_base)
        metrics['cpu_percent'] = max(0, min(100, cpu_adjusted * (1 + random.gauss(0, server.volatility))))
        
        # Memory with baseline adjustment
        mem_base = baseline['memory_percent']
        mem_min, mem_max = timeframe.memory_range
        mem_adjusted = mem_base * random.uniform(mem_min/mem_base, mem_max/mem_base)
        metrics['memory_percent'] = max(0, min(100, mem_adjusted * (1 + random.gauss(0, server.volatility))))
        
        # Disk usage (slower changing)
        disk_base = baseline['disk_percent']
        disk_min, disk_max = timeframe.disk_range
        disk_adjusted = disk_base * random.uniform(disk_min/disk_base, disk_max/disk_base)
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
        
        # Availability (1.0 for online servers)
        metrics['availability'] = 1.0
        
        # Determine status and severity
        is_anomaly = (random.random() < timeframe.anomaly_probability or 
                     metrics['cpu_percent'] > 90 or 
                     metrics['memory_percent'] > 95 or 
                     metrics['java_heap_usage'] > 95)
        
        status = 'anomaly' if is_anomaly else 'normal'
        
        # Determine severity based on metrics
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
    
    def generate_dataset(self, total_hours: int = 168, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate complete metrics dataset over specified time period."""
        logger.info(f"🚀 Starting dataset generation for {total_hours} hours ({total_hours/24:.1f} days)")
        
        # Generate timeframe sequence
        timeframe_sequence = self._generate_timeframe_sequence(total_hours)
        logger.info(f"📅 Generated timeframe sequence: {len(timeframe_sequence)} periods")
        
        # Log sequence summary
        sequence_summary = {}
        for timeframe_name, duration in timeframe_sequence:
            sequence_summary[timeframe_name] = sequence_summary.get(timeframe_name, 0) + duration
        
        for name, total_minutes in sequence_summary.items():
            percentage = (total_minutes / (total_hours * 60)) * 100
            logger.info(f"   {name}: {total_minutes} minutes ({percentage:.1f}%)")
        
        # Generate metrics for each server and timeframe
        current_time = self.generation_start_time
        total_samples = 0
        
        for timeframe_name, duration_minutes in timeframe_sequence:
            timeframe = self.timeframes[timeframe_name]
            samples_in_period = duration_minutes * self.samples_per_minute
            
            logger.info(f"🔄 Generating {samples_in_period} samples for {timeframe_name} "
                       f"({duration_minutes} minutes)")
            
            for minute in range(duration_minutes):
                for sample in range(self.samples_per_minute):
                    sample_time = current_time + timedelta(
                        minutes=minute, 
                        seconds=sample * self.poll_interval_seconds
                    )
                    
                    # Generate metrics for each server at this timestamp
                    for server in self.servers:
                        metrics_sample = self._generate_metrics_for_timeframe(
                            server, timeframe, sample_time
                        )
                        self.training_samples.append(metrics_sample)
                        total_samples += 1
            
            current_time += timedelta(minutes=duration_minutes)
        
        # Create final dataset
        dataset = {
            'training_samples': self.training_samples,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'generation_start_time': self.generation_start_time.isoformat(),
                'total_samples': total_samples,
                'total_hours': total_hours,
                'total_days': total_hours / 24,
                'servers_count': len(self.servers),
                'poll_interval_seconds': self.poll_interval_seconds,
                'samples_per_minute': self.samples_per_minute,
                'timeframe_sequence': timeframe_sequence,
                'timeframe_distribution': sequence_summary,
                'anomaly_samples': sum(1 for s in self.training_samples if s['status'] == 'anomaly'),
                'normal_samples': sum(1 for s in self.training_samples if s['status'] == 'normal'),
                'server_profiles': [
                    {
                        'name': s.name,
                        'heavy_usage': s.heavy_usage,
                        'problem_child': s.problem_child,
                        'volatility': s.volatility
                    } for s in self.servers
                ],
                'format_version': '2.0'
            }
        }
        
        # Calculate final statistics
        anomaly_count = dataset['metadata']['anomaly_samples']
        normal_count = dataset['metadata']['normal_samples']
        anomaly_ratio = anomaly_count / total_samples if total_samples > 0 else 0
        
        logger.info(f"✅ Dataset generation completed!")
        logger.info(f"   Total samples: {total_samples:,}")
        logger.info(f"   Normal samples: {normal_count:,} ({(1-anomaly_ratio)*100:.1f}%)")
        logger.info(f"   Anomaly samples: {anomaly_count:,} ({anomaly_ratio*100:.1f}%)")
        logger.info(f"   Servers: {len(self.servers)}")
        logger.info(f"   Time span: {total_hours} hours ({total_hours/24:.1f} days)")
        
        # Save to file if specified
        if output_file:
            self.save_dataset(dataset, output_file)
        
        return dataset
    
    def save_dataset(self, dataset: Dict[str, Any], output_file: str):
        """Save dataset to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving dataset to {output_path}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            
            # Get file size
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Dataset saved successfully ({file_size_mb:.1f} MB)")
            
        except Exception as e:
            logger.error(f"❌ Failed to save dataset: {e}")
            raise


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Enhanced Metrics Dataset Generator for Predictive Monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 1 week of data (default)
  python metrics_generator.py
  
  # Generate 24 hours of data
  python metrics_generator.py --hours 24
  
  # Generate 30 days with custom output
  python metrics_generator.py --hours 720 --output custom_metrics.json
  
  # Quick test with 1 hour
  python metrics_generator.py --hours 1 --output test_metrics.json
  
  # Generate with custom config
  python metrics_generator.py --config my_config.json --hours 168
        """
    )
    
    parser.add_argument(
        '--hours', '-t',
        type=int,
        default=168,  # 1 week default
        help='Total hours of metrics to generate (default: 168 = 1 week)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='enhanced_metrics_dataset.json',
        help='Output file path (default: enhanced_metrics_dataset.json)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Configuration file path (optional)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducible generation'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode (1 hour, minimal servers)'
    )
    
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Preview timeframe sequence without generating full dataset'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set random seed for reproducibility
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        logger.info(f"🎲 Random seed set to {args.seed}")
    
    # Load configuration
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"📖 Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            return 1
    
    # Quick test mode
    if args.quick_test:
        args.hours = 1
        args.output = 'quick_test_metrics.json'
        logger.info("🚀 Quick test mode: 1 hour, minimal dataset")
    
    # Validate arguments
    if args.hours <= 0:
        logger.error("❌ Hours must be positive")
        return 1
    
    if args.hours > 8760:  # More than 1 year
        logger.warning(f"⚠️  Generating {args.hours} hours ({args.hours/24:.1f} days) - this will be a large dataset")
        confirm = input("Continue? (y/N): ")
        if confirm.lower() != 'y':
            return 0
    
    try:
        # Initialize generator
        generator = MetricsDatasetGenerator(config)
        
        # Preview mode
        if args.preview:
            logger.info("🔍 Preview mode: showing timeframe sequence")
            sequence = generator._generate_timeframe_sequence(args.hours)
            
            print(f"\n📅 TIMEFRAME SEQUENCE FOR {args.hours} HOURS:")
            print("=" * 60)
            
            total_minutes = 0
            sequence_summary = {}
            
            for i, (timeframe_name, duration) in enumerate(sequence, 1):
                print(f"{i:3d}. {timeframe_name:15s} - {duration:3d} minutes")
                total_minutes += duration
                sequence_summary[timeframe_name] = sequence_summary.get(timeframe_name, 0) + duration
            
            print("\n📊 SUMMARY:")
            print("-" * 40)
            for name, total_mins in sorted(sequence_summary.items()):
                percentage = (total_mins / total_minutes) * 100
                hours = total_mins / 60
                print(f"{name:15s}: {total_mins:4d} min ({hours:5.1f}h) - {percentage:5.1f}%")
            
            print(f"\nTotal: {total_minutes} minutes ({total_minutes/60:.1f} hours)")
            return 0
        
        # Generate dataset
        logger.info(f"🚀 Starting generation of {args.hours} hours of metrics data")
        start_time = datetime.now()
        
        dataset = generator.generate_dataset(
            total_hours=args.hours,
            output_file=args.output
        )
        
        generation_time = datetime.now() - start_time
        
        # Display final statistics
        print(f"\n🎉 GENERATION COMPLETED!")
        print("=" * 50)
        print(f"Time taken: {generation_time}")
        print(f"Output file: {args.output}")
        print(f"File size: {Path(args.output).stat().st_size / (1024*1024):.1f} MB")
        print(f"Total samples: {dataset['metadata']['total_samples']:,}")
        print(f"Servers: {dataset['metadata']['servers_count']}")
        print(f"Time span: {args.hours} hours ({args.hours/24:.1f} days)")
        print(f"Anomaly ratio: {dataset['metadata']['anomaly_samples'] / dataset['metadata']['total_samples'] * 100:.1f}%")
        
        print(f"\n📊 TIMEFRAME DISTRIBUTION:")
        for name, minutes in dataset['metadata']['timeframe_distribution'].items():
            percentage = (minutes / (args.hours * 60)) * 100
            print(f"  {name:15s}: {percentage:5.1f}%")
        
        print(f"\n💡 Usage:")
        print(f"  python -c \"import json; data=json.load(open('{args.output}')); print(f'Loaded {{len(data[\\\"training_samples\\\"])}} samples')\"")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Generation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        return 1


def create_sample_config():
    """Create a sample configuration file."""
    config = {
        "description": "Enhanced Metrics Dataset Generator Configuration",
        "servers": {
            "production_count": 17,
            "staging_count": 10,
            "compute_count": 15,
            "service_count": 10,
            "container_count": 5
        },
        "timeframes": {
            "idle_probability": 0.25,
            "healthy_probability": 0.35,
            "spike_probability": 0.15,
            "heavy_load_probability": 0.15,
            "critical_probability": 0.05,
            "maintenance_probability": 0.05
        },
        "metrics": {
            "poll_interval_seconds": 5,
            "anomaly_base_probability": 0.1,
            "problem_child_ratio": 0.08,
            "heavy_usage_ratio": 0.3
        },
        "output": {
            "format_version": "2.0",
            "include_server_profiles": True,
            "include_timeframe_info": True
        }
    }
    
    with open('metrics_generator_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Sample configuration created: metrics_generator_config.json")


def validate_dataset(dataset_file: str):
    """Validate generated dataset format and statistics."""
    try:
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
        
        print(f"🔍 DATASET VALIDATION: {dataset_file}")
        print("=" * 50)
        
        # Check structure
        required_keys = ['training_samples', 'metadata']
        missing_keys = [key for key in required_keys if key not in dataset]
        if missing_keys:
            print(f"❌ Missing required keys: {missing_keys}")
            return False
        
        # Check metadata
        metadata = dataset['metadata']
        print(f"✅ Format version: {metadata.get('format_version', 'unknown')}")
        print(f"✅ Total samples: {metadata.get('total_samples', 0):,}")
        print(f"✅ Servers: {metadata.get('servers_count', 0)}")
        print(f"✅ Time span: {metadata.get('total_hours', 0)} hours")
        
        # Check samples
        samples = dataset['training_samples']
        if not samples:
            print("❌ No training samples found")
            return False
        
        # Validate first sample structure
        sample = samples[0]
        required_sample_keys = ['id', 'timestamp', 'server_name', 'status', 'metrics']
        missing_sample_keys = [key for key in required_sample_keys if key not in sample]
        if missing_sample_keys:
            print(f"❌ Sample missing keys: {missing_sample_keys}")
            return False
        
        # Check metrics structure
        metrics = sample['metrics']
        expected_metrics = ['cpu_percent', 'memory_percent', 'disk_percent', 'load_average']
        missing_metrics = [key for key in expected_metrics if key not in metrics]
        if missing_metrics:
            print(f"❌ Sample missing metrics: {missing_metrics}")
            return False
        
        # Statistics
        normal_count = sum(1 for s in samples if s['status'] == 'normal')
        anomaly_count = sum(1 for s in samples if s['status'] == 'anomaly')
        anomaly_ratio = anomaly_count / len(samples) if samples else 0
        
        print(f"✅ Normal samples: {normal_count:,} ({(1-anomaly_ratio)*100:.1f}%)")
        print(f"✅ Anomaly samples: {anomaly_count:,} ({anomaly_ratio*100:.1f}%)")
        
        # Check timestamp ordering
        timestamps = [datetime.fromisoformat(s['timestamp']) for s in samples[:100]]  # Check first 100
        if timestamps != sorted(timestamps):
            print("⚠️  Timestamps may not be properly ordered")
        else:
            print("✅ Timestamps are properly ordered")
        
        print("\n🎉 Dataset validation passed!")
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    # Special commands
    if len(sys.argv) > 1:
        if sys.argv[1] == 'create-config':
            create_sample_config()
            sys.exit(0)
        elif sys.argv[1] == 'validate' and len(sys.argv) > 2:
            success = validate_dataset(sys.argv[2])
            sys.exit(0 if success else 1)
        elif sys.argv[1] == 'help-config':
            print("""
🔧 CONFIGURATION HELP

The generator accepts a JSON configuration file with these sections:

servers:
  production_count: Number of production servers (default: 17)
  staging_count: Number of staging servers (default: 10)
  compute_count: Number of compute servers (default: 15)
  service_count: Number of service servers (default: 10)
  container_count: Number of container servers (default: 5)

timeframes:
  idle_probability: Probability of idle periods (default: 0.25)
  healthy_probability: Probability of healthy periods (default: 0.35)
  setup_probability: Probability of setup periods (default: 0.15)
  heavy_load_probability: Probability of heavy load (default: 0.15)
  critical_probability: Probability of critical issues (default: 0.05)
  maintenance_probability: Probability of maintenance (default: 0.05)

metrics:
  poll_interval_seconds: Seconds between polls (default: 5)
  anomaly_base_probability: Base anomaly probability (default: 0.1)
  problem_child_ratio: Ratio of problem servers (default: 0.08)
  heavy_usage_ratio: Ratio of heavy usage servers (default: 0.3)

Create a sample config with: python metrics_generator.py create-config
            """)
            sys.exit(0)
    
    # Run main function
    sys.exit(main())