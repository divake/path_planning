#!/usr/bin/env python3
"""
STEP 2: NOISE MODEL IMPLEMENTATION
Goal: Add perception noise to obstacles by varying wall thickness
Expected: Walls appear thicker/thinner, narrowing or widening passages
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from Search_based_Planning.Search_2D import env


def identify_wall_segments(obstacles):
    """
    Identify continuous wall segments in the environment.
    Returns dict of wall segments with their orientation.
    """
    walls = {
        'horizontal': [],
        'vertical': [],
        'boundary': set()
    }
    
    # Identify boundary walls (edges of the grid)
    for (x, y) in obstacles:
        if x == 0 or x == 50 or y == 0 or y == 30:
            walls['boundary'].add((x, y))
    
    # Find horizontal walls (continuous in x direction)
    y_groups = {}
    for (x, y) in obstacles:
        if (x, y) not in walls['boundary']:
            if y not in y_groups:
                y_groups[y] = []
            y_groups[y].append(x)
    
    for y, x_list in y_groups.items():
        x_list = sorted(x_list)
        # Find continuous segments
        segment = [x_list[0]]
        for i in range(1, len(x_list)):
            if x_list[i] == x_list[i-1] + 1:
                segment.append(x_list[i])
            else:
                if len(segment) >= 3:  # At least 3 cells to be a wall
                    walls['horizontal'].append((segment[0], segment[-1], y))
                segment = [x_list[i]]
        if len(segment) >= 3:
            walls['horizontal'].append((segment[0], segment[-1], y))
    
    # Find vertical walls (continuous in y direction)
    x_groups = {}
    for (x, y) in obstacles:
        if (x, y) not in walls['boundary']:
            if x not in x_groups:
                x_groups[x] = []
            x_groups[x].append(y)
    
    for x, y_list in x_groups.items():
        y_list = sorted(y_list)
        # Find continuous segments
        segment = [y_list[0]]
        for i in range(1, len(y_list)):
            if y_list[i] == y_list[i-1] + 1:
                segment.append(y_list[i])
            else:
                if len(segment) >= 3:  # At least 3 cells to be a wall
                    walls['vertical'].append((x, segment[0], segment[-1]))
                segment = [y_list[i]]
        if len(segment) >= 3:
            walls['vertical'].append((x, segment[0], segment[-1]))
    
    return walls


def add_perception_noise_thickness(true_obstacles, thickness_std=0.3, seed=None):
    """
    Add perception noise by varying wall thickness (Option A: Uniform).
    
    Args:
        true_obstacles: Set of (x, y) tuples representing true obstacle positions
        thickness_std: Standard deviation for thickness variation
        seed: Random seed for reproducibility
    
    Returns:
        perceived_obstacles: Set with walls of varying thickness
    """
    if seed is not None:
        np.random.seed(seed)
    
    perceived_obstacles = set()
    
    # Identify wall segments
    walls = identify_wall_segments(true_obstacles)
    
    # Always keep boundary walls (just copy them)
    for obs in walls['boundary']:
        perceived_obstacles.add(obs)
    
    # Process horizontal walls with thickness variation
    for (x_start, x_end, y) in walls['horizontal']:
        # Determine thickness change for this wall segment (uniform for entire segment)
        thickness_change = np.random.normal(0, thickness_std)
        thickness_pixels = int(round(abs(thickness_change)))
        
        # Add the original wall
        for x in range(x_start, x_end + 1):
            perceived_obstacles.add((x, y))
        
        # Add thickness perpendicular to wall
        if thickness_pixels > 0:
            # Always thicken (ignore sign of thickness_change for now)
            # Add cells above and below
            for x in range(x_start, x_end + 1):
                for dy in range(1, thickness_pixels + 1):
                    # Add above
                    if 0 < y + dy < 30:
                        perceived_obstacles.add((x, y + dy))
                    # Add below
                    if 0 < y - dy < 30:
                        perceived_obstacles.add((x, y - dy))
    
    # Process vertical walls with thickness variation
    for (x, y_start, y_end) in walls['vertical']:
        # Determine thickness change for this wall segment (uniform for entire segment)
        thickness_change = np.random.normal(0, thickness_std)
        thickness_pixels = int(round(abs(thickness_change)))
        
        # Add the original wall
        for y in range(y_start, y_end + 1):
            perceived_obstacles.add((x, y))
        
        # Add thickness perpendicular to wall
        if thickness_pixels > 0:
            # Always thicken (ignore sign for now)
            # Add cells left and right
            for y in range(y_start, y_end + 1):
                for dx in range(1, thickness_pixels + 1):
                    # Add right
                    if 0 < x + dx < 50:
                        perceived_obstacles.add((x + dx, y))
                    # Add left
                    if 0 < x - dx < 50:
                        perceived_obstacles.add((x - dx, y))
        
        # SPECIAL: Extend walls into passages to narrow them
        # This is key for making passages appear narrower
        extension = int(round(np.random.normal(0, thickness_std * 2)))
        if extension > 0:
            # Extend wall beyond its original endpoints
            for extra_y in range(1, extension + 1):
                # Extend upward
                if y_end + extra_y < 30:
                    perceived_obstacles.add((x, y_end + extra_y))
                # Extend downward
                if y_start - extra_y > 0:
                    perceived_obstacles.add((x, y_start - extra_y))
    
    # Add any isolated obstacles that aren't part of walls
    isolated = true_obstacles - walls['boundary']
    for obs in isolated:
        # Check if it's part of a wall segment
        is_wall_part = False
        for (x_start, x_end, y) in walls['horizontal']:
            if obs[1] == y and x_start <= obs[0] <= x_end:
                is_wall_part = True
                break
        if not is_wall_part:
            for (x, y_start, y_end) in walls['vertical']:
                if obs[0] == x and y_start <= obs[1] <= y_end:
                    is_wall_part = True
                    break
        
        # If not part of wall, add with slight random thickness
        if not is_wall_part:
            perceived_obstacles.add(obs)
            if np.random.random() < 0.3:  # 30% chance to thicken isolated obstacles
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        new_x, new_y = obs[0] + dx, obs[1] + dy
                        if 0 < new_x < 50 and 0 < new_y < 30:
                            perceived_obstacles.add((new_x, new_y))
    
    return perceived_obstacles


def analyze_passage_widths(true_obstacles, perceived_obstacles):
    """
    Analyze how passages have changed between true and perceived.
    """
    # Check specific passage points
    passage_points = [
        (20, 15),  # First narrow passage
        (30, 15),  # Second narrow passage  
        (40, 15),  # Third narrow passage
    ]
    
    results = {}
    for px, py in passage_points:
        # Find gap width in true environment
        true_gap = 0
        for y in range(py - 10, py + 10):
            if (px, y) not in true_obstacles:
                true_gap += 1
        
        # Find gap width in perceived environment
        perceived_gap = 0
        for y in range(py - 10, py + 10):
            if (px, y) not in perceived_obstacles:
                perceived_gap += 1
        
        results[f"x={px}"] = {
            'true_gap': true_gap,
            'perceived_gap': perceived_gap,
            'change': perceived_gap - true_gap
        }
    
    return results


def visualize_noise_effect(thickness_std=0.3):
    """
    Visualize the effect of thickness-based noise on obstacle perception.
    """
    # Get true obstacles
    true_env = env.Env()
    true_obstacles = true_env.obs
    
    # Create multiple noise instances
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Thickness-Based Noise Model (σ_thickness = {thickness_std})', fontsize=16)
    
    # Different noise levels to show
    noise_levels = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]
    
    for idx, (ax, noise) in enumerate(zip(axes.flat, noise_levels)):
        # Add noise
        if noise == 0.0:
            perceived = true_obstacles
            ax.set_title('Ground Truth (No Noise)')
        else:
            perceived = add_perception_noise_thickness(true_obstacles, thickness_std=noise, seed=42+idx)
            ax.set_title(f'Perceived with σ = {noise}')
        
        # Set up plot
        ax.set_xlim(-1, 51)
        ax.set_ylim(-1, 31)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Plot true obstacles (light gray)
        for obs in true_obstacles:
            rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                    linewidth=1, edgecolor='gray',
                                    facecolor='lightgray', alpha=0.5)
            ax.add_patch(rect)
        
        # Plot perceived obstacles (black)
        for obs in perceived:
            rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                    linewidth=1, edgecolor='black',
                                    facecolor='black', alpha=0.8)
            ax.add_patch(rect)
        
        # Highlight narrow passages
        for x in [20, 30, 40]:
            ax.axvline(x, color='red', alpha=0.2, linestyle='--')
        
        # Analyze passage changes
        passage_analysis = analyze_passage_widths(true_obstacles, perceived)
        
        # Add passage width info
        info_text = f"Obstacles: {len(perceived)}\n"
        info_text += f"Passage widths:\n"
        for passage, data in passage_analysis.items():
            change = data['change']
            if change < 0:
                info_text += f"  {passage}: {data['perceived_gap']} (narrower by {abs(change)})\n"
            elif change > 0:
                info_text += f"  {passage}: {data['perceived_gap']} (wider by {change})\n"
            else:
                info_text += f"  {passage}: {data['perceived_gap']} (unchanged)\n"
        
        ax.text(2, 28, info_text, fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                verticalalignment='top')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('results', exist_ok=True)
    save_path = 'results/step2_noise_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nNoise visualization saved to: {save_path}")
    plt.close()


def test_noise_model():
    """
    Test the thickness-based noise model.
    """
    print("="*50)
    print("STEP 2: THICKNESS-BASED NOISE MODEL TEST")
    print("="*50)
    
    # Get environment
    true_env = env.Env()
    true_obstacles = true_env.obs
    
    print(f"\nTrue environment has {len(true_obstacles)} obstacles")
    
    # Identify walls
    walls = identify_wall_segments(true_obstacles)
    print(f"\nWall segments identified:")
    print(f"  Horizontal walls: {len(walls['horizontal'])}")
    print(f"  Vertical walls: {len(walls['vertical'])}")
    print(f"  Boundary cells: {len(walls['boundary'])}")
    
    # Test different noise levels
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    print("\nTesting different thickness noise levels:")
    print("-" * 40)
    
    for thickness_std in noise_levels:
        # Run multiple times to get statistics
        obstacle_counts = []
        passage_narrowing = []
        
        for i in range(100):
            perceived = add_perception_noise_thickness(true_obstacles, thickness_std, seed=i)
            obstacle_counts.append(len(perceived))
            
            # Check passage widths
            passage_analysis = analyze_passage_widths(true_obstacles, perceived)
            avg_narrowing = np.mean([data['change'] for data in passage_analysis.values()])
            passage_narrowing.append(avg_narrowing)
        
        avg_obstacles = np.mean(obstacle_counts)
        std_obstacles = np.std(obstacle_counts)
        avg_narrowing = np.mean(passage_narrowing)
        
        print(f"\nσ_thickness = {thickness_std:3.1f}:")
        print(f"  Avg obstacles: {avg_obstacles:6.1f} ± {std_obstacles:4.1f}")
        print(f"  Increase: {(avg_obstacles - len(true_obstacles)):+6.1f} obstacles")
        print(f"  Avg passage change: {avg_narrowing:+4.2f} cells")
        if avg_narrowing < 0:
            print(f"  → Passages are NARROWER (good!)")
        elif avg_narrowing > 0:
            print(f"  → Passages are WIDER (not desired)")
    
    print("\n" + "="*50)
    print("NARROW PASSAGE ANALYSIS")
    print("="*50)
    
    # Detailed passage analysis
    thickness_std = 0.3  # Recommended value
    print(f"\nUsing σ_thickness = {thickness_std}")
    
    # Test multiple runs
    narrowed_count = 0
    blocked_count = 0
    
    for trial in range(100):
        perceived = add_perception_noise_thickness(true_obstacles, thickness_std, seed=trial)
        passage_analysis = analyze_passage_widths(true_obstacles, perceived)
        
        for passage, data in passage_analysis.items():
            if data['change'] < 0:
                narrowed_count += 1
            if data['perceived_gap'] == 0:
                blocked_count += 1
    
    print(f"\nOut of 300 passage measurements (100 trials × 3 passages):")
    print(f"  Narrowed: {narrowed_count} ({narrowed_count/3:.0f}%)")
    print(f"  Completely blocked: {blocked_count} ({blocked_count/3:.0f}%)")
    
    # Generate visualization
    print("\nGenerating thickness-based noise visualization...")
    visualize_noise_effect(thickness_std)
    
    print("\n" + "="*50)
    print("SUCCESS CRITERIA CHECK")
    print("="*50)
    
    # Run a single test with different seed
    perceived = add_perception_noise_thickness(true_obstacles, thickness_std=0.3, seed=456)
    
    success = True
    
    # Check walls are intact
    walls_intact = True
    for (x, y) in true_obstacles:
        if x == 0 or x == 50 or y == 0 or y == 30:  # Boundary walls
            if (x, y) not in perceived:
                walls_intact = False
                break
    
    if walls_intact:
        print("✓ Boundary walls intact")
    else:
        print("✗ Boundary walls broken")
        success = False
    
    # Check obstacle count increased (thickening)
    if len(perceived) >= len(true_obstacles):
        print(f"✓ Obstacles preserved/thickened ({len(true_obstacles)} → {len(perceived)})")
    else:
        print(f"✗ Obstacles lost ({len(true_obstacles)} → {len(perceived)})")
        success = False
    
    # Check passages affected
    passage_analysis = analyze_passage_widths(true_obstacles, perceived)
    passages_affected = any(data['change'] != 0 for data in passage_analysis.values())
    
    if passages_affected:
        print("✓ Passages affected by noise")
    else:
        print("✗ Passages unchanged")
        success = False
    
    if success:
        print("\n✅ READY TO PROCEED TO STEP 3")
        print("   (Verify that narrowed passages cause naive method to fail)")
    else:
        print("\n❌ Issues found - debug before proceeding")
    
    return success


if __name__ == "__main__":
    # Test the noise model
    success = test_noise_model()
    
    if success:
        print("\nNext step: Verify problem with naive method (step3_verify_problem.py)")
    else:
        print("\nDebug noise model before moving to Step 3")