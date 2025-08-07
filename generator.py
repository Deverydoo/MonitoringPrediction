#!/usr/bin/env python3
"""
generator.py - Metrics Dataset Generator
Generates realistic server metrics for TFT training
"""

import json
import random
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from config import CONFIG


class MetricsGenerator:
    """Generate realistic server metrics dataset."""
    
    def __init__(self):
        self.servers = self._create_server_profiles()
    
    def _create_server_profiles(self) -> List[Dict]:
        """Create diverse server profiles."""
        servers = []
        server_count = CONFIG["servers_count"]
        
        for i in range(server_count):
            servers.append({
                "name": f"server-{i:03d}",
                "baseline_cpu": random.uniform(20, 40),
                "baseline_memory": random.uniform(40, 60),
                "baseline_disk": random.uniform(30, 50),
                "baseline_load": random.uniform(1.0, 2.5),
                "volatility": random.uniform(0.1, 0.2),
                "problem_prone": random.random() < 0.1,
            })
        
        return servers
    
    def generate(self, hours: int, output_file: Optional[str] = None) -> bool:
        """Generate metrics dataset."""
        print(f"🚀 Generating {hours} hours of metrics data...")
        
        samples = []
        start_time = datetime.now() - timedelta(hours=hours)
        poll_interval = CONFIG["poll_interval_minutes"]
        
        time_points = (hours * 60) // poll_interval
        total_samples = time_points * len(self.servers)
        print(f"📊 Expected samples: {total_samples:,}")
        
        for time_offset in range(0, hours * 60, poll_interval):
            timestamp = start_time + timedelta(minutes=time_offset)
            
            # Determine load pattern based on hour
            hour = timestamp.hour
            if 6 <= hour < 9:
                load_mult = 1.5  # Morning spike
            elif 9 <= hour < 17:
                load_mult = 1.0  # Normal business
            elif 17 <= hour < 20:
                load_mult = 1.3  # Evening peak
            else:
                load_mult = 0.4  # Idle
            
            for server in self.servers:
                metrics = self._generate_metrics(server, load_mult)
                
                # Determine if anomaly
                is_anomaly = (
                    random.random() < CONFIG["anomaly_ratio"] or
                    metrics["cpu_percent"] > 90 or
                    metrics["memory_percent"] > 95
                )
                
                sample = {
                    "id": f"{server['name']}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                    "timestamp": timestamp.isoformat(),
                    "server_name": server["name"],
                    "status": "anomaly" if is_anomaly else "normal",
                    "metrics": metrics,
                    "severity": self._get_severity(metrics),
                }
                
                samples.append(sample)
        
        # Create dataset
        dataset = {
            "training_samples": samples,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_samples": len(samples),
                "time_span_hours": hours,
                "servers_count": len(self.servers),
                "anomaly_ratio": sum(1 for s in samples if s["status"] == "anomaly") / len(samples),
                "poll_interval_seconds": poll_interval * 60,
            }
        }
        
        # Save dataset
        if output_file is None:
            output_file = str(Path(CONFIG["training_dir"]) / "metrics_dataset.json")
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"✅ Generated {len(samples):,} samples")
        print(f"💾 Saved to: {output_path}")
        
        return True
    
    def _generate_metrics(self, server: Dict, load_mult: float) -> Dict:
        """Generate realistic metrics for a server."""
        volatility = server["volatility"]
        
        cpu = server["baseline_cpu"] * load_mult * (1 + random.gauss(0, volatility))
        memory = server["baseline_memory"] * load_mult * (1 + random.gauss(0, volatility))
        
        if server["problem_prone"] and random.random() < 0.2:
            cpu *= 1.4
            memory *= 1.3
        
        return {
            "cpu_percent": max(0, min(100, cpu)),
            "memory_percent": max(0, min(100, memory)),
            "disk_percent": max(0, min(100, server["baseline_disk"] + random.gauss(0, 3))),
            "load_average": max(0, server["baseline_load"] * load_mult + random.gauss(0, 0.2)),
        }
    
    def _get_severity(self, metrics: Dict) -> str:
        """Determine severity level."""
        if metrics["cpu_percent"] > 95 or metrics["memory_percent"] > 98:
            return "critical"
        elif metrics["cpu_percent"] > 85 or metrics["memory_percent"] > 90:
            return "high"
        elif metrics["cpu_percent"] > 70 or metrics["memory_percent"] > 80:
            return "medium"
        return "low"


# Module interface
def generate_dataset(hours: int, output_file: Optional[str] = None) -> bool:
    """Generate dataset - module interface."""
    generator = MetricsGenerator()
    return generator.generate(hours, output_file)


def main():
    """CLI interface."""
    parser = argparse.ArgumentParser(description="Generate server metrics dataset")
    parser.add_argument("--hours", type=int, default=168, help="Hours of data")
    parser.add_argument("--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    success = generate_dataset(args.hours, args.output)
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())