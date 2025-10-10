#!/usr/bin/env python3
"""
run_demo.py - Convenient Demo Runner
Generate demo data and run dashboard in one command
"""

import argparse
import sys
from pathlib import Path

from demo_data_generator import generate_demo_dataset
from tft_dashboard_refactored import run_dashboard


def run_demo(duration_minutes: int = 5,
            fleet_size: int = 10,
            seed: int = 42,
            scenario: str = 'degrading',
            tick_interval_seconds: int = 5,
            refresh_seconds: int = 30,
            output_dir: str = "./demo_data/",
            regenerate: bool = False):
    """
    Run complete demo: generate data and launch dashboard.

    Args:
        duration_minutes: Duration of demo data
        fleet_size: Number of servers
        seed: Random seed for reproducibility
        scenario: Health scenario - 'healthy', 'degrading', or 'critical'
        tick_interval_seconds: How often to ingest new data (simulates real-time)
        refresh_seconds: How often to refresh dashboard visualizations
        output_dir: Where to store demo data
        regenerate: Force regeneration of demo data
    """

    demo_path = Path(output_dir) / "demo_dataset.parquet"

    # Check if demo data exists
    if demo_path.exists() and not regenerate:
        print(f"✅ Demo data already exists: {demo_path}")
        print(f"   (Use --regenerate to create new data)")
    else:
        print("📊 Generating demo dataset...")
        print(f"   Scenario: {scenario.upper()}")
        success = generate_demo_dataset(
            output_dir=output_dir,
            duration_minutes=duration_minutes,
            fleet_size=fleet_size,
            seed=seed,
            scenario=scenario
        )

        if not success:
            print("❌ Failed to generate demo data")
            return 1

    # Run dashboard
    print("\n🚀 Launching demo dashboard...")
    print(f"   Data source: {demo_path}")
    print(f"   Tick interval: {tick_interval_seconds}s (data ingestion)")
    print(f"   Refresh interval: {refresh_seconds}s (visualization)")
    print("   Press Ctrl+C to stop\n")

    try:
        run_dashboard(
            data_path=str(demo_path),
            data_format='parquet',
            tick_interval_sec=tick_interval_seconds,
            refresh_sec=refresh_seconds,
            save_plots=False
        )
        return 0
    except KeyboardInterrupt:
        print("\n✅ Demo stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Run TFT Demo: Generate data and launch dashboard with configurable scenarios",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--duration", type=int, default=5,
                       help="Demo duration in minutes")
    parser.add_argument("--fleet-size", type=int, default=10,
                       help="Number of servers in fleet")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--scenario", type=str, default='degrading',
                       choices=['healthy', 'degrading', 'critical'],
                       help="Health scenario: healthy (no issues), degrading (gradual), critical (acute spikes)")
    parser.add_argument("--tick-interval", type=int, default=5,
                       help="Data ingestion interval in seconds (simulates real-time arrival)")
    parser.add_argument("--refresh", type=int, default=30,
                       help="Dashboard refresh interval in seconds")
    parser.add_argument("--output-dir", default="./demo_data/",
                       help="Directory for demo data")
    parser.add_argument("--regenerate", action="store_true",
                       help="Force regeneration of demo data")

    args = parser.parse_args()

    return run_demo(
        duration_minutes=args.duration,
        fleet_size=args.fleet_size,
        seed=args.seed,
        scenario=args.scenario,
        tick_interval_seconds=args.tick_interval,
        refresh_seconds=args.refresh,
        output_dir=args.output_dir,
        regenerate=args.regenerate
    )


if __name__ == "__main__":
    sys.exit(main())
