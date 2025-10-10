#!/usr/bin/env python3
"""
Test all three demo scenarios

IMPORTANT: Run this in the py310 environment:
    conda activate py310
    python test_scenarios.py
"""

from demo_data_generator import generate_demo_dataset
import shutil
from pathlib import Path

def test_all_scenarios():
    """Test healthy, degrading, and critical scenarios."""

    # Clean test directory
    test_dir = Path('./test_scenarios/')
    if test_dir.exists():
        shutil.rmtree(test_dir)

    results = {}

    # Test HEALTHY scenario
    print('=' * 60)
    print('Testing HEALTHY scenario...')
    print('=' * 60)
    success = generate_demo_dataset(
        output_dir='./test_scenarios/healthy/',
        duration_minutes=1,
        fleet_size=5,
        seed=42,
        scenario='healthy'
    )
    results['healthy'] = success
    print(f'\n✅ HEALTHY: PASSED' if success else '\n❌ HEALTHY: FAILED')

    # Test DEGRADING scenario
    print('\n' + '=' * 60)
    print('Testing DEGRADING scenario...')
    print('=' * 60)
    success = generate_demo_dataset(
        output_dir='./test_scenarios/degrading/',
        duration_minutes=1,
        fleet_size=5,
        seed=42,
        scenario='degrading'
    )
    results['degrading'] = success
    print(f'\n✅ DEGRADING: PASSED' if success else '\n❌ DEGRADING: FAILED')

    # Test CRITICAL scenario
    print('\n' + '=' * 60)
    print('Testing CRITICAL scenario...')
    print('=' * 60)
    success = generate_demo_dataset(
        output_dir='./test_scenarios/critical/',
        duration_minutes=1,
        fleet_size=5,
        seed=42,
        scenario='critical'
    )
    results['critical'] = success
    print(f'\n✅ CRITICAL: PASSED' if success else '\n❌ CRITICAL: FAILED')

    # Summary
    print('\n' + '=' * 60)
    print('SCENARIO TEST SUMMARY')
    print('=' * 60)
    for scenario, passed in results.items():
        status = '✅ PASSED' if passed else '❌ FAILED'
        print(f'{scenario.upper():12s}: {status}')

    all_passed = all(results.values())
    print('\n' + ('🎉 ALL TESTS PASSED!' if all_passed else '❌ SOME TESTS FAILED'))

    return all_passed

if __name__ == '__main__':
    test_all_scenarios()
