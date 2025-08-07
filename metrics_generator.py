#!/usr/bin/env python3
"""
metrics_generator.py - Server Metrics Dataset Generator
Generates realistic server metrics for TFT training
Supports both module import and command-line usage
"""

import os
import json
import random
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass

# Import config
try:
    from config import CONFIG
except ImportError:
    CONFIG = {"time_span_hours": 168, "servers_count": 57, "anomaly_ratio": 0.15}


@dataclass
class ServerProfile:
    """Server baseline characteristics."""
    name: str
    baseline_cpu: float = 25.0
    baseline_memory: float = 45.0
    baseline_disk: float = 35.0
    baseline_load: float = 1.2
    volatility: float = 0.15
    problem_child: bool = False
    heavy_usage: bool = False


class MetricsGenerator:
    """Generate realistic server metrics dataset."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or CONFIG
        self.poll_interval_minutes = 5
        self.servers = self._create_server_profiles()
        
    def _create_server_profiles(self) -> List[ServerProfile]:
        """Create diverse server profiles."""
        servers = []
        server_count = self.config.get("servers_count", 57)
        
        # Production servers (30% of fleet)
        prod_count = int(server_count * 0.3)
        for i in range(prod_count):
            servers.append(ServerProfile(
                name=f"prod-{i:03d}",
                baseline_cpu=35.0,
                baseline_memory=60.0,
                baseline_disk=45.0,
                baseline_load=2.5,
                heavy_usage=True,
                problem_child=(random.random() < 0.1)
            ))
        
        # Staging servers (20% of fleet)
        staging_count = int(server_count * 0.2)
        for i in range(staging_count):
            servers.append(ServerProfile(
                name=f"staging-{i:03d}",
                baseline_cpu=20.0,
                baseline_memory=40.0,
                baseline_disk=30.0,
                baseline_load=1.0,
                problem_child=(random.random() < 0.05)
            ))
        
        # Service servers (remaining)
        service_count = server_count - prod_count - staging_count
        for i in range(service_count):
            servers.append(ServerProfile(
                name=f"service-{i:03d}",
                baseline_cpu=15.0,
                baseline_memory=35.0,
                baseline_disk=25.0,
                baseline_load=0.8,
                volatility=0.1
            ))
        
        return servers
    
    def generate(self, hours: Optional[int] = None, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate metrics dataset.
        
        Args:
            hours: Hours of data to generate (uses config if None)
            output_file: Output path (uses default if None)
            
        Returns:
            Generated dataset dictionary
        """
        # Use provided hours or fall back to config
        total_hours = hours if hours is not None else self.config.get("time_span_hours", 168)
        
        print(f"🚀 Generating {total_hours} hours of metrics data")
        print(f"📊 Servers: {len(self.servers)}, Poll interval: {self.poll_interval_minutes} min")
        
        # Calculate total samples
        time_points = (total_hours * 60) // self.poll_interval_minutes
        total_samples = time_points * len(self.servers)
        print(f"🎯 Expected samples: {total_samples:,}")
        
        # Generate samples
        training_samples = []
        start_time = datetime.now() - timedelta(hours=total_hours)
        
        for time_offset in range(0, total_hours * 60, self.poll_interval_minutes):
            timestamp = start_time + timedelta(minutes=time_offset)
            
            # Determine timeframe based on hour
            hour = timestamp.hour
            if 0 <= hour < 6:
                timeframe = "idle"
                cpu_mult, mem_mult = 0.3, 0.4
            elif 6 <= hour < 9:
                timeframe = "morning_spike"
                cpu_mult, mem_mult = 1.5, 1.4
            elif 9 <= hour < 17:
                timeframe = "healthy"
                cpu_mult, mem_mult = 1.0, 1.0
            elif 17 <= hour < 20:
                timeframe = "heavy_load"
                cpu_mult, mem_mult = 1.3, 1.2
            else:
                timeframe = "idle"
                cpu_mult, mem_mult = 0.5, 0.6
            
            # Generate metrics for each server
            for server in self.servers:
                metrics = self._generate_metrics(server, cpu_mult, mem_mult)
                
                # Determine if anomaly
                is_anomaly = (
                    random.random() < self.config.get("anomaly_ratio", 0.15) or
                    metrics["cpu_percent"] > 90 or
                    metrics["memory_percent"] > 95
                )
                
                sample = {
                    "id": f"{server.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                    "timestamp": timestamp.isoformat(),
                    "server_name": server.name,
                    "status": "anomaly" if is_anomaly else "normal",
                    "timeframe": timeframe,
                    "metrics": metrics,
                    "severity": self._get_severity(metrics),
                    "explanation": f"Server {server.name} in {timeframe} state"
                }
                
                training_samples.append(sample)
        
        # Create dataset
        dataset = {
            "training_samples": training_samples,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_samples": len(training_samples),
                "time_span_hours": total_hours,
                "servers_count": len(self.servers),
                "anomaly_ratio": sum(1 for s in training_samples if s["status"] == "anomaly") / len(training_samples),
                "poll_interval_seconds": self.poll_interval_minutes * 60,
                "format_version": "2.0",
                "enhanced": True
            }
        }
        
        # Save to file
        if output_file is None:
            output_file = str(Path(self.config.get("training_dir", "./training")) / "metrics_dataset.json")
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"✅ Generated {len(training_samples):,} samples")
        print(f"💾 Saved to: {output_path}")
        
        return dataset
    
    def _generate_metrics(self, server: ServerProfile, cpu_mult: float, mem_mult: float) -> Dict[str, float]:
        """Generate metrics for a server."""
        # Apply multipliers and add noise
        cpu = server.baseline_cpu * cpu_mult * (1 + random.gauss(0, server.volatility))
        memory = server.baseline_memory * mem_mult * (1 + random.gauss(0, server.volatility))
        
        if server.heavy_usage:
            cpu *= 1.3
            memory *= 1.2
        
        if server.problem_child and random.random() < 0.3:
            cpu *= 1.5
            memory *= 1.4
        
        return {
            "cpu_percent": max(0, min(100, cpu)),
            "memory_percent": max(0, min(100, memory)),
            "disk_percent": max(0, min(100, server.baseline_disk + random.gauss(0, 5))),
            "load_average": max(0, server.baseline_load * cpu_mult + random.gauss(0, 0.3)),
            "network_bytes_sent": max(0, 1000000 * cpu_mult + random.gauss(0, 100000)),
            "network_bytes_recv": max(0, 800000 * cpu_mult + random.gauss(0, 80000)),
            "disk_read_bytes": max(0, 500000 * cpu_mult + random.gauss(0, 50000)),
            "disk_write_bytes": max(0, 300000 * cpu_mult + random.gauss(0, 30000)),
            "java_heap_usage": max(0, min(100, 55 * mem_mult + random.gauss(0, 10))),
            "java_gc_time": max(0, 2.0 * cpu_mult + random.gauss(0, 0.5))
        }
    
    def _get_severity(self, metrics: Dict[str, float]) -> str:
        """Determine severity level."""
        if metrics["cpu_percent"] > 95 or metrics["memory_percent"] > 98:
            return "critical"
        elif metrics["cpu_percent"] > 85 or metrics["memory_percent"] > 90:
            return "high"
        elif metrics["cpu_percent"] > 70 or metrics["memory_percent"] > 80:
            return "medium"
        else:
            return "low"


# Module interface
def generate_dataset(hours: Optional[int] = None, output_file: Optional[str] = None, config: Optional[Dict] = None) -> Dict[str, Any]:
    """Generate metrics dataset - module interface."""
    generator = MetricsGenerator(config)
    return generator.generate(hours, output_file)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Generate server metrics dataset for TFT training")
    parser.add_argument("--hours", type=int, help="Hours of data to generate (overrides config)")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--servers", type=int, help="Number of servers")
    parser.add_argument("--anomaly-ratio", type=float, help="Ratio of anomalies (0-1)")
    
    args = parser.parse_args()
    
    # Override config if args provided
    config = CONFIG.copy()
    if args.servers:
        config["servers_count"] = args.servers
    if args.anomaly_ratio:
        config["anomaly_ratio"] = args.anomaly_ratio
    
    # Generate dataset
    generator = MetricsGenerator(config)
    generator.generate(args.hours, args.output)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())