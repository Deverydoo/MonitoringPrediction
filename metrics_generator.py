#!/usr/bin/env python3
"""
metrics_generator.py - FIXED Data Generator with Consistent Keys
Generates data in the exact format that the trainer expects
"""

import json
import random
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class ServerMetricsGenerator:
    """Generate time series data with consistent keys for TFT training."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            "servers_count": 10,
            "poll_interval_minutes": 5,
            "anomaly_ratio": 0.15
        }
        
    def generate(self, hours: int, output_file: Optional[str] = None) -> bool:
        """Generate dataset with EXACT keys that trainer expects."""
        print(f"🚀 Generating {hours} hours of time series data...")
        
        time_points = (hours * 60) // self.config["poll_interval_minutes"]
        start_time = datetime.now() - timedelta(hours=hours)
        
        print(f"📊 This will create ~{time_points * self.config['servers_count']:,} samples")
        
        # Generate records in EXACT format trainer expects
        all_records = []
        
        for server_id in range(self.config["servers_count"]):
            server_name = f"server-{server_id:03d}"
            
            # Realistic baseline values
            baseline = {
                'cpu_percent': random.uniform(20, 50),
                'memory_percent': random.uniform(30, 60),
                'disk_percent': random.uniform(25, 45),
                'load_average': random.uniform(0.5, 2.0)
            }
            
            for time_idx in range(time_points):
                timestamp = start_time + timedelta(minutes=time_idx * self.config["poll_interval_minutes"])
                
                # Generate realistic metrics with temporal patterns
                metrics = self._generate_realistic_metrics(baseline, time_idx, timestamp)
                
                # Create record in EXACT format trainer expects
                record = {
                    'id': f"{server_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                    'timestamp': timestamp.isoformat(),
                    'server_id': server_name,  # trainer expects server_id, not server_name
                    'status': 'anomaly' if random.random() < self.config["anomaly_ratio"] else 'normal',
                    'timeframe': 'normal',
                    'severity': 'low',
                    'explanation': 'Generated sample',
                    # Flatten metrics to top level (trainer expects flat structure)
                    **metrics  # This spreads the metrics as top-level keys
                }
                
                all_records.append(record)
        
        # Create dataset with EXACT keys trainer expects
        dataset = {
            'records': all_records,  # trainer expects 'records' key
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_samples': len(all_records),
                'time_span_hours': hours,
                'servers_count': self.config["servers_count"],
                'poll_interval_seconds': self.config["poll_interval_minutes"] * 60,
                'anomaly_ratio': self.config["anomaly_ratio"],
                'format_version': "2.0"
            }
        }
        
        # Save to file
        output_path = Path(output_file or './training/metrics_dataset.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"✅ Generated {len(all_records):,} records")
        print(f"💾 Saved to: {output_path}")
        
        # Verify structure matches trainer expectations
        print(f"📋 Dataset structure:")
        print(f"   Records: {len(all_records)}")
        print(f"   Servers: {dataset['metadata']['servers_count']}")
        print(f"   Time span: {dataset['metadata']['time_span_hours']} hours")
        if all_records:
            print(f"   Sample keys: {list(all_records[0].keys())}")
        
        return True
    
    def _generate_realistic_metrics(self, baseline: Dict, time_idx: int, timestamp: datetime) -> Dict:
        """Generate realistic metrics with temporal patterns."""
        hour = timestamp.hour
        
        # Time-based load multipliers
        if 2 <= hour <= 6:  # Night hours
            load_mult = 0.3
        elif 7 <= hour <= 9:  # Morning peak
            load_mult = 1.5
        elif 10 <= hour <= 16:  # Business hours
            load_mult = 1.0
        elif 17 <= hour <= 20:  # Evening peak
            load_mult = 1.3
        else:  # Off hours
            load_mult = 0.6
        
        metrics = {}
        for metric, base_value in baseline.items():
            # Apply time-based variations
            value = base_value * load_mult * (1 + random.gauss(0, 0.15))
            
            # Add gradual trends
            trend = 0.01 * time_idx / 100
            value += trend
            
            # Add cyclical patterns
            if metric == 'cpu_percent':
                cycle = 5 * np.sin(2 * np.pi * time_idx / 12)  # 1-hour cycle
                value += cycle
            elif metric == 'memory_percent':
                # Memory leak simulation
                value += time_idx * 0.001
            
            metrics[metric] = value
        
        return self._bound_metrics(metrics)
    
    def _bound_metrics(self, metrics: Dict) -> Dict:
        """Apply realistic bounds to metrics."""
        bounded = {}
        
        for metric, value in metrics.items():
            if metric.endswith('_percent'):
                bounded[metric] = max(0, min(100, value))
            elif metric == 'load_average':
                bounded[metric] = max(0, min(20, value))
            else:
                bounded[metric] = max(0, value)
        
        return bounded


def generate_dataset(hours: int = 24, output_file: Optional[str] = None) -> bool:
    """Module interface for generating dataset."""
    generator = ServerMetricsGenerator()
    return generator.generate(hours, output_file)


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Generate server metrics dataset")
    parser.add_argument("--hours", type=int, default=24, help="Hours of data to generate")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--servers", type=int, default=10, help="Number of servers")
    
    args = parser.parse_args()
    
    config = {
        "servers_count": args.servers,
        "poll_interval_minutes": 5,
        "anomaly_ratio": 0.15
    }
    
    generator = ServerMetricsGenerator(config)
    success = generator.generate(args.hours, args.output)
    
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())