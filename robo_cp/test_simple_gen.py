#!/usr/bin/env python3
"""Simple test to isolate the issue"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../python_motion_planning/src'))

# Test if we can import the module
print("1. Importing module...")
from dataset_generation.generate_dataset_improved import ImprovedDatasetGenerator
print("   Success!")

# Create generator with minimal samples
print("2. Creating generator...")
gen = ImprovedDatasetGenerator(num_samples=2, robot_radius=0.5)
print("   Success!")

# Try to generate
print("3. Calling generate_dataset()...")
try:
    features, labels, metadata = gen.generate_dataset()
    print(f"   Success! Generated {len(features)} samples")
except Exception as e:
    print(f"   Failed: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")