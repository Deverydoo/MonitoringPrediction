#!/usr/bin/env python3
"""
Quick test to verify TFT model loading works.
"""
import sys
from pathlib import Path

print("="*60)
print("Testing TFT Model Loading")
print("="*60)

# Import the TFTInference class
try:
    from tft_inference import TFTInference
    print("[OK] tft_inference module imported")
except Exception as e:
    print(f"[ERROR] Failed to import: {e}")
    sys.exit(1)

# Try to load the model
try:
    print("\nAttempting to load TFT model...")
    inference = TFTInference(model_path=None, use_real_model=True)

    if inference.use_real_model and inference.model is not None:
        print("\n[SUCCESS] TFT model loaded successfully!")
        print(f"  Model directory: {inference.model_dir}")
        print(f"  Device: {inference.device}")
        print(f"  Model type: {inference.config.get('model_type', 'Unknown')}")

        # Count parameters
        if inference.model:
            num_params = sum(p.numel() for p in inference.model.parameters())
            print(f"  Parameters: {num_params:,}")

        sys.exit(0)
    else:
        print("\n[WARNING] Model did not load - running in heuristic mode")
        sys.exit(2)

except Exception as e:
    print(f"\n[ERROR] Exception during model loading:")
    print(f"  {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
