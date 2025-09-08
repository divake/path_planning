#!/usr/bin/env python3
"""
Parse and visualize all 7 MRPB maps to see their actual dimensions and layouts
"""

import numpy as np
import matplotlib.pyplot as plt
from mrpb_map_parser import MRPBMapParser
import matplotlib.patches as patches


def parse_and_visualize_all_maps():
    """Parse all 7 MRPB maps and create comprehensive visualization"""
    
    # The 7 maps we're interested in
    map_names = [
        'office01add',      # Indoor office
        'office02',         # Larger office  
        'shopping_mall',    # Mall environment
        'room02',           # Family house
        'maze',             # Challenging maze
        'track',            # U-shaped track
        'narrow_graph'      # Acute angles
    ]
    
    # Parse all maps and collect info
    all_maps = {}
    dimensions_summary = []
    
    print("="*80)
    print("PARSING ALL MRPB MAPS - ACTUAL DIMENSIONS AND OBSTACLES")
    print("="*80)
    
    for map_name in map_names:
        try:
            print(f"\n{map_name.upper()}")
            print("-"*40)
            
            parser = MRPBMapParser(map_name, '../mrpb_dataset')
            info = parser.get_environment_info()
            all_maps[map_name] = parser
            
            # Print detailed info
            print(f"  Actual dimensions: {info['width_meters']:.1f} x {info['height_meters']:.1f} meters")
            print(f"  Number of obstacles: {info['num_obstacles']}")
            print(f"  Robot radius: {info['robot_radius']} meters")
            print(f"  Resolution: {info['resolution']} m/pixel")
            print(f"  Origin: {info['origin']}")
            
            dimensions_summary.append({
                'name': map_name,
                'width': info['width_meters'],
                'height': info['height_meters'],
                'obstacles': info['num_obstacles'],
                'area': info['width_meters'] * info['height_meters']
            })
            
        except Exception as e:
            print(f"  ERROR: Could not parse {map_name}: {e}")
    
    # Print summary table
    print("\n" + "="*80)
    print("DIMENSIONS SUMMARY TABLE")
    print("="*80)
    print(f"{'Map Name':<15} {'Width (m)':<12} {'Height (m)':<12} {'Area (m²)':<12} {'Obstacles':<10}")
    print("-"*70)
    
    for dim in dimensions_summary:
        print(f"{dim['name']:<15} {dim['width']:<12.1f} {dim['height']:<12.1f} "
              f"{dim['area']:<12.1f} {dim['obstacles']:<10}")
    
    print("\n" + "="*80)
    print("ROBOT SPECIFICATIONS")
    print("="*80)
    print(f"Robot radius: 0.17 meters")
    print(f"Robot diameter: 0.34 meters")
    print(f"This is the actual MRPB robot size - we will use this exactly!")
    
    return all_maps


def create_comprehensive_visualization(all_maps):
    """Create a single figure showing all 7 maps with their actual layouts"""
    
    # Create figure with subplots for all 7 maps
    fig = plt.figure(figsize=(20, 12))
    
    # Define grid layout - 2 rows, 4 columns
    positions = [
        (0, 0), (0, 1), (0, 2), (0, 3),  # First row
        (1, 0), (1, 1), (1, 2)           # Second row
    ]
    
    map_order = ['office01add', 'office02', 'shopping_mall', 'room02', 
                 'maze', 'track', 'narrow_graph']
    
    for idx, map_name in enumerate(map_order):
        if map_name not in all_maps:
            continue
            
        parser = all_maps[map_name]
        row, col = positions[idx]
        
        # Create subplot
        ax = plt.subplot2grid((2, 4), (row, col))
        
        # Draw obstacles
        for obs in parser.obstacles:
            x, y, w, h = obs
            rect = patches.Rectangle((x, y), w, h,
                                    linewidth=0.5, 
                                    edgecolor='black',
                                    facecolor='gray',
                                    alpha=0.8)
            ax.add_patch(rect)
        
        # Draw robot size reference at origin
        robot_circle = patches.Circle((0, 0), parser.robot_radius,
                                     linewidth=2, 
                                     edgecolor='green',
                                     facecolor='none',
                                     linestyle='--',
                                     label=f'Robot r={parser.robot_radius}m')
        ax.add_patch(robot_circle)
        
        # Draw scale reference (1 meter bar)
        scale_x = parser.origin[0] + 1
        scale_y = parser.origin[1] + parser.height_meters - 1
        ax.plot([scale_x, scale_x + 1], [scale_y, scale_y], 
                'r-', linewidth=3, label='1 meter')
        
        # Set limits and labels
        ax.set_xlim(parser.origin[0], parser.origin[0] + parser.width_meters)
        ax.set_ylim(parser.origin[1], parser.origin[1] + parser.height_meters)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2, linewidth=0.5)
        
        # Title with dimensions
        ax.set_title(f'{map_name}\n{parser.width_meters:.1f} × {parser.height_meters:.1f} m, '
                    f'{len(parser.obstacles)} obstacles',
                    fontsize=10, fontweight='bold')
        
        # Add axis labels for corner plots
        if row == 1:
            ax.set_xlabel('X (meters)', fontsize=8)
        if col == 0:
            ax.set_ylabel('Y (meters)', fontsize=8)
        
        # Legend only for first plot
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
        
        # Tick labels
        ax.tick_params(labelsize=8)
    
    # Overall title
    fig.suptitle('MRPB Dataset - All 7 Maps with Actual Dimensions and Layouts\n'
                 'Using Original Robot Size: radius = 0.17m',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.25, wspace=0.15)
    
    # Save high-resolution figure
    plt.savefig('mrpb_all_maps_actual.png', dpi=200, bbox_inches='tight')
    print(f"\nVisualization saved to: mrpb_all_maps_actual.png")
    
    plt.show()


def create_individual_visualizations(all_maps):
    """Create individual high-quality visualizations for each map"""
    
    for map_name, parser in all_maps.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Occupancy grid
        ax1.imshow(parser.occupancy_grid, cmap='gray', origin='lower')
        ax1.set_title(f'{map_name}: Original Occupancy Grid')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        ax1.grid(True, alpha=0.3)
        
        # Right: Extracted obstacles
        ax2.set_aspect('equal')
        
        # Draw obstacles
        for obs in parser.obstacles:
            x, y, w, h = obs
            rect = patches.Rectangle((x, y), w, h,
                                    linewidth=1, 
                                    edgecolor='darkred',
                                    facecolor='lightcoral',
                                    alpha=0.7)
            ax2.add_patch(rect)
        
        # Add robot at different positions to show scale
        test_positions = [
            (parser.origin[0] + 2, parser.origin[1] + 2),
            (0, 0),
            (parser.origin[0] + parser.width_meters/2, 
             parser.origin[1] + parser.height_meters/2)
        ]
        
        colors = ['green', 'blue', 'orange']
        labels = ['Start example', 'Origin', 'Center']
        
        for pos, color, label in zip(test_positions, colors, labels):
            if (parser.origin[0] <= pos[0] <= parser.origin[0] + parser.width_meters and
                parser.origin[1] <= pos[1] <= parser.origin[1] + parser.height_meters):
                robot_circle = patches.Circle(pos, parser.robot_radius,
                                             linewidth=2, 
                                             edgecolor=color,
                                             facecolor='none',
                                             label=f'{label} (r={parser.robot_radius}m)')
                ax2.add_patch(robot_circle)
        
        # Set limits
        ax2.set_xlim(parser.origin[0] - 0.5, parser.origin[0] + parser.width_meters + 0.5)
        ax2.set_ylim(parser.origin[1] - 0.5, parser.origin[1] + parser.height_meters + 0.5)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        
        # Title with full info
        ax2.set_title(f'{map_name}: Extracted Obstacles\n'
                     f'Size: {parser.width_meters:.1f} × {parser.height_meters:.1f} meters, '
                     f'Obstacles: {len(parser.obstacles)}')
        ax2.set_xlabel('X (meters)')
        ax2.set_ylabel('Y (meters)')
        
        plt.suptitle(f'MRPB Map: {map_name.upper()}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save individual map
        filename = f'mrpb_{map_name}_detailed.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved detailed view: {filename}")
        
        plt.close()  # Close to avoid too many open figures


def analyze_map_complexity(all_maps):
    """Analyze complexity metrics for each map"""
    
    print("\n" + "="*80)
    print("MAP COMPLEXITY ANALYSIS")
    print("="*80)
    
    metrics = []
    
    for map_name, parser in all_maps.items():
        # Calculate free space
        total_pixels = parser.occupancy_grid.size
        occupied_pixels = np.sum(parser.occupancy_grid == 100)
        free_pixels = np.sum(parser.occupancy_grid == 0)
        unknown_pixels = np.sum(parser.occupancy_grid == 50)
        
        free_ratio = free_pixels / total_pixels
        occupied_ratio = occupied_pixels / total_pixels
        
        # Calculate average obstacle size
        if parser.obstacles:
            avg_obstacle_area = np.mean([w*h for x, y, w, h in parser.obstacles])
        else:
            avg_obstacle_area = 0
        
        metrics.append({
            'name': map_name,
            'free_space': free_ratio * 100,
            'occupied': occupied_ratio * 100,
            'avg_obs_area': avg_obstacle_area,
            'num_obstacles': len(parser.obstacles)
        })
        
        print(f"\n{map_name}:")
        print(f"  Free space: {free_ratio*100:.1f}%")
        print(f"  Occupied space: {occupied_ratio*100:.1f}%")
        print(f"  Unknown space: {unknown_pixels/total_pixels*100:.1f}%")
        print(f"  Average obstacle area: {avg_obstacle_area:.2f} m²")
        print(f"  Number of obstacles: {len(parser.obstacles)}")
    
    # Rank by difficulty (less free space = harder)
    metrics_sorted = sorted(metrics, key=lambda x: x['free_space'])
    
    print("\n" + "="*80)
    print("DIFFICULTY RANKING (based on free space)")
    print("="*80)
    print(f"{'Rank':<6} {'Map Name':<15} {'Free Space %':<15} {'Difficulty':<15}")
    print("-"*60)
    
    for i, m in enumerate(metrics_sorted, 1):
        if m['free_space'] < 70:
            difficulty = "HARD"
        elif m['free_space'] < 80:
            difficulty = "MEDIUM"
        else:
            difficulty = "EASY"
        
        print(f"{i:<6} {m['name']:<15} {m['free_space']:<15.1f} {difficulty:<15}")


if __name__ == "__main__":
    # Parse all maps
    all_maps = parse_and_visualize_all_maps()
    
    # Create comprehensive visualization
    if all_maps:
        print("\nCreating comprehensive visualization...")
        create_comprehensive_visualization(all_maps)
        
        print("\nCreating individual detailed visualizations...")
        create_individual_visualizations(all_maps)
        
        # Analyze complexity
        analyze_map_complexity(all_maps)
        
        print("\n" + "="*80)
        print("COMPLETE! All MRPB maps have been parsed and visualized.")
        print("Check the generated PNG files for visualizations.")
        print("="*80)