#!/usr/bin/env python3
"""
Server Name Encoding/Decoding Utility

Provides deterministic hash-based encoding for server names to support
dynamic server fleets in production environments.

Conforms to: DATA_CONTRACT.md v1.0.0
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple


class ServerEncoder:
    """
    Hash-based server name encoder with bidirectional mapping.

    Uses SHA256 hashing to create deterministic numeric IDs from server names.
    Maintains mapping table for decoding predictions back to server names.
    """

    def __init__(self, mapping_file: Path = None):
        """
        Initialize encoder.

        Args:
            mapping_file: Optional path to existing mapping JSON
        """
        self.name_to_id: Dict[str, str] = {}
        self.id_to_name: Dict[str, str] = {}

        if mapping_file and Path(mapping_file).exists():
            self.load_mapping(mapping_file)

    @staticmethod
    def encode_server_name(server_name: str) -> str:
        """
        Create deterministic hash-based encoding for a server name.

        Args:
            server_name: Original hostname (e.g., 'ppvra00a0018')

        Returns:
            Consistent numeric string ID (e.g., '123456')

        Examples:
            >>> ServerEncoder.encode_server_name('ppvra00a0018')
            '849234'
            >>> ServerEncoder.encode_server_name('ppvra00a0018')
            '849234'  # Same input -> same output
        """
        # Use first 8 chars of SHA256 hash as numeric ID
        hash_obj = hashlib.sha256(server_name.encode('utf-8'))
        hash_int = int(hash_obj.hexdigest()[:8], 16)
        # Keep it reasonable for TFT embedding layer (6 digits max)
        return str(hash_int % 1_000_000)

    def create_mapping(self, server_names: List[str]) -> None:
        """
        Create bidirectional mapping for list of server names.

        Args:
            server_names: List of unique server hostnames

        Side Effects:
            Updates self.name_to_id and self.id_to_name
        """
        # Create name -> ID mapping
        self.name_to_id = {
            name: self.encode_server_name(name)
            for name in server_names
        }

        # Create reverse ID -> name mapping
        self.id_to_name = {v: k for k, v in self.name_to_id.items()}

        # Check for hash collisions
        if len(self.id_to_name) != len(self.name_to_id):
            collisions = len(self.name_to_id) - len(self.id_to_name)
            print(f"[WARNING] {collisions} hash collision(s) detected!")
            print("   Consider increasing hash space or using different algorithm")

    def encode(self, server_name: str) -> str:
        """
        Encode server name to ID using existing mapping.

        Args:
            server_name: Server hostname

        Returns:
            Encoded server ID

        Raises:
            KeyError: If server_name not in mapping (call create_mapping first)
        """
        if server_name not in self.name_to_id:
            # Auto-encode new servers on-the-fly
            encoded = self.encode_server_name(server_name)
            self.name_to_id[server_name] = encoded
            self.id_to_name[encoded] = server_name
            print(f"[INFO] Auto-encoded new server: {server_name} -> {encoded}")

        return self.name_to_id[server_name]

    def decode(self, server_id: str) -> str:
        """
        Decode server ID back to original name.

        Args:
            server_id: Encoded server ID

        Returns:
            Original server name or 'UNKNOWN_{id}' if not in mapping
        """
        return self.id_to_name.get(server_id, f'UNKNOWN_{server_id}')

    def save_mapping(self, output_file: Path) -> None:
        """
        Save mapping to JSON file.

        Args:
            output_file: Path to output JSON file

        File Format:
            {
                "name_to_id": {"ppvra00a0018": "123456", ...},
                "id_to_name": {"123456": "ppvra00a0018", ...},
                "version": "1.0.0"
            }
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        mapping_data = {
            'name_to_id': self.name_to_id,
            'id_to_name': self.id_to_name,
            'version': '1.0.0',
            'total_servers': len(self.name_to_id)
        }

        with open(output_file, 'w') as f:
            json.dump(mapping_data, f, indent=2)

        print(f"[OK] Server mapping saved to: {output_file}")
        print(f"   Total servers: {len(self.name_to_id)}")

    def load_mapping(self, input_file: Path) -> None:
        """
        Load mapping from JSON file.

        Args:
            input_file: Path to mapping JSON file

        Side Effects:
            Updates self.name_to_id and self.id_to_name
        """
        input_file = Path(input_file)

        if not input_file.exists():
            raise FileNotFoundError(f"Mapping file not found: {input_file}")

        with open(input_file, 'r') as f:
            mapping_data = json.load(f)

        self.name_to_id = mapping_data.get('name_to_id', {})
        self.id_to_name = mapping_data.get('id_to_name', {})

        version = mapping_data.get('version', 'unknown')
        total = mapping_data.get('total_servers', len(self.name_to_id))

        print(f"[OK] Server mapping loaded from: {input_file}")
        print(f"   Version: {version}")
        print(f"   Total servers: {total}")

    def get_stats(self) -> Dict:
        """
        Get statistics about current mapping.

        Returns:
            Dict with mapping statistics
        """
        return {
            'total_servers': len(self.name_to_id),
            'total_ids': len(self.id_to_name),
            'has_collisions': len(self.name_to_id) != len(self.id_to_name),
            'sample_mappings': dict(list(self.name_to_id.items())[:5])
        }


def validate_encoding_stability(server_names: List[str], iterations: int = 1000) -> bool:
    """
    Validate that encoding is stable across multiple calls.

    Args:
        server_names: List of server names to test
        iterations: Number of encoding iterations per server

    Returns:
        True if all encodings are stable
    """
    print(f"\n[TEST] Validating encoding stability ({iterations} iterations)...")

    for server_name in server_names:
        first_encoding = ServerEncoder.encode_server_name(server_name)

        for i in range(iterations):
            current_encoding = ServerEncoder.encode_server_name(server_name)
            if current_encoding != first_encoding:
                print(f"[ERROR] Unstable encoding for {server_name}")
                print(f"   Expected: {first_encoding}")
                print(f"   Got: {current_encoding}")
                return False

    print(f"[OK] All encodings stable across {iterations} iterations")
    return True


def detect_collisions(server_names: List[str]) -> Tuple[bool, List[Tuple[str, str]]]:
    """
    Detect hash collisions in a list of server names.

    Args:
        server_names: List of server names to check

    Returns:
        (has_collisions, list_of_collision_pairs)
    """
    print(f"\n[TEST] Checking for hash collisions in {len(server_names)} servers...")

    encoder = ServerEncoder()
    encoder.create_mapping(server_names)

    collisions = []
    seen_ids = {}

    for name in server_names:
        encoded_id = encoder.encode(name)
        if encoded_id in seen_ids and seen_ids[encoded_id] != name:
            collisions.append((seen_ids[encoded_id], name))
        else:
            seen_ids[encoded_id] = name

    if collisions:
        print(f"[WARNING] Found {len(collisions)} collision(s):")
        for name1, name2 in collisions:
            print(f"   {name1} <-> {name2}")
        return True, collisions
    else:
        print(f"[OK] No collisions detected")
        return False, []


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("SERVER ENCODER - Test Suite")
    print("=" * 60)

    # Test 1: Basic encoding
    print("\n[TEST 1] Basic Encoding")
    test_servers = [
        'ppvra00a0018',
        'ppvra00a0025',
        'ppvra00a0028',
        'ppvra00a0030',
        'pprva00a01',
        'psrva00a01',
        'cppr01'
    ]

    encoder = ServerEncoder()
    encoder.create_mapping(test_servers)

    print("\nServer Name -> ID Mappings:")
    for name, server_id in encoder.name_to_id.items():
        print(f"   {name:20s} -> {server_id}")

    # Test 2: Encoding stability
    validate_encoding_stability(test_servers[:3], iterations=100)

    # Test 3: Collision detection
    detect_collisions(test_servers)

    # Test 4: Round-trip encoding/decoding
    print("\n[TEST 4] Round-trip Encoding/Decoding")
    for server_name in test_servers[:3]:
        encoded = encoder.encode(server_name)
        decoded = encoder.decode(encoded)
        status = "[OK]" if decoded == server_name else "[ERROR]"
        print(f"   {status} {server_name} -> {encoded} -> {decoded}")

    # Test 5: Save/Load mapping
    print("\n[TEST 5] Save/Load Mapping")
    test_file = Path("test_server_mapping.json")
    encoder.save_mapping(test_file)

    new_encoder = ServerEncoder(test_file)
    if new_encoder.name_to_id == encoder.name_to_id:
        print("[OK] Mapping saved and loaded successfully")
    else:
        print("[ERROR] Mapping mismatch after load")

    # Cleanup
    if test_file.exists():
        test_file.unlink()
        print("[OK] Cleaned up test file")

    # Test 6: Statistics
    print("\n[TEST 6] Encoder Statistics")
    stats = encoder.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
