#!/usr/bin/env python3
"""
API Key Generator and Manager for ArgusAI

Built by Craig Giannelli and Claude Code

This script generates a secure API key and stores it in a dedicated file
(.nordiq_key) separate from other environment variables (.env).

Usage:
    python bin/generate_api_key.py              # Generate new key if none exists
    python bin/generate_api_key.py --force      # Force regenerate even if exists
    python bin/generate_api_key.py --show       # Show current key without regenerating
"""

import os
import sys
import secrets
import string
from pathlib import Path

# Get the NordIQ root directory (parent of bin/)
NORDIQ_ROOT = Path(__file__).parent.parent


def generate_secure_key(length: int = 64) -> str:
    """Generate a cryptographically secure random API key."""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def read_existing_key() -> str:
    """Read existing API key from .nordiq_key if it exists."""
    key_file = NORDIQ_ROOT / '.nordiq_key'
    if key_file.exists():
        try:
            with open(key_file, 'r') as f:
                key = f.read().strip()
                return key
        except Exception as e:
            print(f"[WARNING] Error reading existing key: {e}")
    return ""


def write_nordiq_key(api_key: str):
    """Write API key to .nordiq_key file."""
    key_file = NORDIQ_ROOT / '.nordiq_key'

    with open(key_file, 'w') as f:
        f.write(api_key)

    # Set file permissions to 600 (owner read/write only) on Unix systems
    try:
        os.chmod(key_file, 0o600)
    except Exception:
        pass  # Windows doesn't support chmod

    print(f"[OK] NordIQ API key: .nordiq_key")


def write_secrets_toml(api_key: str):
    """Write API key to .env file for dashboard."""
    # Legacy Streamlit config no longer used
    secrets_dir.mkdir(exist_ok=True)

    secrets_file = secrets_dir / 'secrets.toml'

    content = f'''# API Key Configuration
# This file contains sensitive configuration values
# DO NOT commit this file to version control!

[daemon]
# API key for TFT Inference Daemon authentication
# This key is also stored in .nordiq_key
api_key = "{api_key}"

# Production deployment notes:
# 1. This key is automatically synced with .nordiq_key
# 2. Set as environment variable: export NORDIQ_API_KEY="{api_key}"
# 3. The start scripts will automatically load from .nordiq_key
'''

    with open(secrets_file, 'w') as f:
        f.write(content)

    print(f"[OK] Dashboard configuration: .env file")


def ensure_gitignore():
    """Ensure .gitignore protects secret files."""
    gitignore_file = NORDIQ_ROOT / '.gitignore'

    entries_needed = [
        '.env file',
        '.nordiq_key',
        '.env'  # Still protect .env in case it contains other secrets
    ]

    if gitignore_file.exists():
        with open(gitignore_file, 'r') as f:
            existing = f.read()

        needs_update = False
        for entry in entries_needed:
            if entry not in existing:
                needs_update = True
                break

        if needs_update:
            with open(gitignore_file, 'a') as f:
                f.write('\n# API Keys and Secrets\n')
                for entry in entries_needed:
                    if entry not in existing:
                        f.write(f'{entry}\n')
            print(f"[OK] Updated .gitignore to protect secrets")
    else:
        with open(gitignore_file, 'w') as f:
            f.write('# API Keys and Secrets\n')
            for entry in entries_needed:
                f.write(f'{entry}\n')
        print(f"[OK] Created .gitignore to protect secrets")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate and manage API keys')
    parser.add_argument('--force', action='store_true',
                       help='Force regenerate even if key exists')
    parser.add_argument('--show', action='store_true',
                       help='Show current key without regenerating')
    args = parser.parse_args()

    # Show current key if requested
    if args.show:
        existing_key = read_existing_key()
        if existing_key:
            print(f"Current API Key: {existing_key}")
        else:
            print("No API key found. Run without --show to generate one.")
        return

    # Check if key already exists
    existing_key = read_existing_key()

    if existing_key and not args.force:
        print("API key already configured!")
        print(f"Key: {existing_key[:20]}...{existing_key[-10:]}")
        print()
        print("The API key is stored in:")
        print("  - .nordiq_key (primary)")
        print("  - .env file (dashboard)")
        print()
        print("To regenerate, run with --force flag")
        return

    # Generate new key
    if args.force and existing_key:
        print("Regenerating API key (--force flag set)...")
    else:
        print("Generating new API key...")

    api_key = generate_secure_key(64)

    # Write to both locations
    write_nordiq_key(api_key)
    write_secrets_toml(api_key)
    ensure_gitignore()

    print()
    print("=" * 60)
    print("API Key Configuration Complete!")
    print("=" * 60)
    print()
    print(f"Generated API Key: {api_key}")
    print()
    print("The API key has been configured in:")
    print("  [OK] .nordiq_key (primary key file)")
    print("  [OK] .env file (dashboard)")
    print()
    print("Production usage:")
    print("  # Set as environment variable:")
    print(f"  export NORDIQ_API_KEY=\"{api_key}\"")
    print()
    print("  # Or start scripts will automatically load from .nordiq_key")
    print("  ./start_all.sh")
    print()
    print("Security notes:")
    print("  - .nordiq_key is separate from .env (won't overwrite other tokens)")
    print("  - Protected by .gitignore (not committed)")
    print("  - Keep this key secret and secure")
    print("  - Rotate periodically for security")
    print()


if __name__ == '__main__':
    main()
