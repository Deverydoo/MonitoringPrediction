#!/usr/bin/env python3
"""
Pre-compile all Python modules to bytecode for faster dashboard startup.

Usage:
    python precompile.py

This recursively compiles all .py files in the project to .pyc bytecode files,
stored in __pycache__/ directories. Subsequent dashboard loads will be faster.
"""

import py_compile
import os
from pathlib import Path


def compile_directory(directory: str = "."):
    """Recursively compile all .py files in directory."""
    compiled_count = 0
    error_count = 0

    print(f"🔧 Pre-compiling Python modules in: {directory}")
    print("-" * 60)

    for root, dirs, files in os.walk(directory):
        # Skip certain directories
        skip_dirs = {'.git', '.venv', 'venv', 'env', '__pycache__',
                     'node_modules', '.conda', 'training', 'models'}
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    py_compile.compile(filepath, doraise=True)
                    compiled_count += 1
                    print(f"✅ Compiled: {filepath}")
                except py_compile.PyCompileError as e:
                    error_count += 1
                    print(f"❌ Error: {filepath} - {e}")

    print("-" * 60)
    print(f"✅ Successfully compiled: {compiled_count} files")
    if error_count > 0:
        print(f"❌ Errors: {error_count} files")
    print("\n🚀 Dashboard startup will now be faster!")


if __name__ == "__main__":
    compile_directory(".")
