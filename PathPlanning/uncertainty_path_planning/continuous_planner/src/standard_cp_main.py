#!/usr/bin/env python3
"""
Standard CP Main Implementation
Complete Standard Conformal Prediction for MRPB path planning

This is the main entry point that coordinates:
1. Environment loading (reuses existing MRPBMapParser)
2. Noise simulation (StandardCPNoiseModel)
3. Nonconformity computation (StandardCPNonconformity)
4. Calibration and path planning
5. Evaluation against naive baseline
"""

import numpy as np
import yaml
import json
import time
import os
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import sys

# Import existing infrastructure
sys.path.append('.')
sys.path.append('../../../CurvesGenerator')
from mrpb_map_parser import MRPBMapParser
from rrt_star_grid_planner import RRTStarGrid  
from mrpb_metrics import MRPBMetrics, NavigationData
from test_runner import generate_realistic_trajectory
from visualization import MRPBVisualizer

# Import our new Standard CP modules
from standard_cp_noise_model import StandardCPNoiseModel
from standard_cp_nonconformity import StandardCPNonconformity


class StandardCPPlanner:
    """
    Main Standard CP implementation that integrates with existing MRPB infrastructure
    
    Provides:
    - Global τ calibration across multiple environments
    - Path planning with safety margins  
    - Evaluation against naive baseline
    - Results compatible with existing visualization
    """
    
    def __init__(self, config_path: str = "standard_cp_config.yaml"):
        """
        Initialize Standard CP planner
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.setup_directories()
        
        # Initialize components
        self.noise_model = StandardCPNoiseModel(config_path)
        self.nonconformity_calc = StandardCPNonconformity(config_path)
        
        # Calibration state
        self.tau = None
        self.calibration_data = {}
        self.calibration_completed = False
        
        # Results storage
        self.results = {
            'calibration': {},
            'validation': {},
            'evaluation': {}
        }
        
        self.logger.info("StandardCP Planner initialized")
        self.logger.info(f"Config: {config_path}")
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration with inheritance from config_env.yaml"""
        try:
            # Load Standard CP specific config
            with open(config_path, 'r') as f:
                cp_config = yaml.safe_load(f)
            
            # Load base configuration (config_env.yaml)
            base_config_file = cp_config.get('base_config_file', 'config_env.yaml')
            try:
                with open(base_config_file, 'r') as f:
                    base_config = yaml.safe_load(f)
                
                # Merge configurations: base config + CP-specific overrides
                merged_config = {}
                merged_config.update(base_config)  # Start with base
                
                # Add CP-specific sections
                for key, value in cp_config.items():
                    if key != 'base_config_file':  # Don't include the import directive
                        merged_config[key] = value
                
                # Use robot config from base, but allow CP overrides
                if 'robot' in base_config:
                    merged_config['robot'] = base_config['robot']
                
                return merged_config
                
            except FileNotFoundError:
                self.logger.warning(f"Base config file not found: {base_config_file}, using CP config only")
                return cp_config
            
        except Exception as e:
            print(f"Error loading config: {e}")
            raise
    
    def setup_logging(self):
        """Setup logging based on configuration"""
        log_level = getattr(logging, self.config['debug']['log_level'])
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Console handler
        if self.config['debug']['log_to_console']:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config['debug']['log_to_file']:
            log_dir = os.path.join(self.config['output']['base_dir'], 
                                  self.config['output']['subdirs']['logs'])
            os.makedirs(log_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime(self.config['output']['timestamp_format'])
            log_file = os.path.join(log_dir, f"standard_cp_{timestamp}.log")
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def setup_directories(self):
        """Create output directories"""
        base_dir = self.config['output']['base_dir']
        
        for subdir in self.config['output']['subdirs'].values():
            dir_path = os.path.join(base_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)
            
        self.logger.debug("Output directories created")
    
    def get_test_environments(self) -> List[Dict]:
        """
        Get environments for testing based on configuration
        
        Returns list of environment configurations for calibration/evaluation
        """
        if self.config['environments']['full_environments']['enabled']:
            # Use full environment set
            envs = []
            for env_config in self.config['environments']['full_environments']['calibration_envs']:
                envs.append(env_config)
            return envs
        else:
            # Use test environments
            return self.config['environments']['test_environments']
    
    def load_environment(self, env_name: str) -> Tuple[MRPBMapParser, Dict]:
        """
        Load MRPB environment using existing infrastructure
        
        Args:
            env_name: Environment name (e.g., 'office01add')
            
        Returns:
            (parser, env_config) tuple
        """
        try:
            # Use existing MRPBMapParser
            parser = MRPBMapParser(env_name)
            
            # Load environment configuration from existing config_env.yaml
            with open('config_env.yaml', 'r') as f:
                mrpb_config = yaml.safe_load(f)
            
            env_config = mrpb_config['environments'][env_name]
            env_config['parser'] = parser
            
            self.logger.debug(f"Loaded environment: {env_name}")
            return parser, env_config
            
        except Exception as e:
            self.logger.error(f"Failed to load environment {env_name}: {e}")
            raise
    
    def plan_naive_path(self, 
                       env_name: str, 
                       test_id: int, 
                       perceived_grid: Optional[np.ndarray] = None,
                       seed: int = 42) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path using naive RRT* (no safety margins)
        
        Reuses existing RRT* infrastructure for consistency
        """
        try:
            parser, env_config = self.load_environment(env_name)
            
            # Get test configuration
            test_config = env_config['tests'][test_id - 1]
            start = test_config['start']
            goal = test_config['goal']
            
            # Use perceived grid or true grid
            if perceived_grid is None:
                perceived_grid = parser.occupancy_grid
            
            # Use existing RRT* planner with timeout protection
            planner = RRTStarGrid(
                start=start,
                goal=goal,
                occupancy_grid=perceived_grid,
                origin=parser.origin,
                resolution=parser.resolution,
                robot_radius=self.config['robot']['radius'],
                max_iter=self.config['planning']['rrt_star']['max_iterations'],
                seed=seed
            )
            
            # Plan with timeout
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Planning timeout")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.config['planning']['timeout']['planning_timeout']))
            
            try:
                path = planner.plan()
                signal.alarm(0)  # Cancel timeout
            except (TimeoutError, Exception) as e:
                signal.alarm(0)  # Cancel timeout
                self.logger.warning(f"Planning failed or timed out: {e}")
                path = None
            finally:
                signal.signal(signal.SIGALRM, old_handler)
            
            if path:
                # Apply quintic polynomial smoothing (reuse existing function)
                try:
                    trajectory_data = generate_realistic_trajectory(
                        path, 
                        parser,
                        max_vel=self.config['planning']['smoothing']['max_velocity'],
                        max_acc=self.config['planning']['smoothing']['max_acceleration']
                    )
                    # Extract just the x,y coordinates from NavigationData
                    smooth_path = [(data.x, data.y) for data in trajectory_data]
                    return smooth_path
                except Exception as e:
                    self.logger.warning(f"Smoothing failed, using raw path: {e}")
                    return path
            
            return None
            
        except Exception as e:
            self.logger.error(f"Naive planning failed for {env_name}-{test_id}: {e}")
            return None
    
    def calibrate_global_tau(self) -> float:
        """
        Calibrate global τ across all test environments
        
        This is the core of Standard CP - single τ for all environments
        """
        self.logger.info("Starting global τ calibration...")
        
        calibration_config = self.config['conformal_prediction']['calibration']
        noise_levels = self.config['noise_model']['noise_levels']
        
        all_scores = []
        detailed_results = {}
        
        # Get environments for calibration
        test_environments = self.get_test_environments()
        
        total_trials = 0
        for env_config in test_environments:
            env_name = env_config['name']
            test_ids = env_config['test_ids']
            
            self.logger.info(f"Calibrating on {env_name} (tests: {test_ids})")
            
            env_scores = []
            env_details = []
            
            # Load environment
            parser, mrpb_env_config = self.load_environment(env_name)
            true_grid = parser.occupancy_grid
            
            for test_id in test_ids:
                for trial in range(calibration_config['trials_per_environment']):
                    # Use deterministic seed sequence
                    trial_seed = calibration_config['random_seed_base'] + total_trials
                    
                    # Select noise level
                    noise_level = noise_levels[trial % len(noise_levels)]
                    
                    # Add realistic perception noise
                    perceived_grid = self.noise_model.add_realistic_noise(
                        true_grid, noise_level=noise_level, seed=trial_seed
                    )
                    
                    # Plan path on perceived environment
                    path = self.plan_naive_path(
                        env_name, test_id, perceived_grid, seed=trial_seed + 1000
                    )
                    
                    # Compute nonconformity score
                    if path:
                        score = self.nonconformity_calc.compute_nonconformity_score(
                            true_grid, perceived_grid, path, 
                            path_id=f"{env_name}-{test_id}-{trial}"
                        )
                    else:
                        # Planning failure penalty
                        score = self.nonconformity_calc.nc_config['penalty_planning_failure']
                    
                    all_scores.append(score)
                    env_scores.append(score)
                    
                    env_details.append({
                        'trial': trial,
                        'test_id': test_id,
                        'noise_level': noise_level,
                        'score': score,
                        'path_found': path is not None,
                        'path_length': len(path) if path else 0
                    })
                    
                    total_trials += 1
                    
                    # More frequent progress updates for small trials
                    progress_interval = 5 if calibration_config.get('fast_mode', False) else 20
                    if total_trials % progress_interval == 0:
                        print(f"  Progress: {total_trials} trials completed...")
                        self.logger.debug(f"Completed {total_trials} calibration trials...")
            
            # Store per-environment results
            detailed_results[env_name] = {
                'scores': env_scores,
                'details': env_details,
                'mean_score': np.mean(env_scores),
                'std_score': np.std(env_scores),
                'success_rate': sum(1 for d in env_details if d['path_found']) / len(env_details)
            }
        
        # Compute global τ
        confidence_level = self.config['conformal_prediction']['confidence_level']
        sorted_scores = sorted(all_scores)
        
        quantile_idx = int(np.ceil((len(sorted_scores) + 1) * confidence_level)) - 1
        quantile_idx = min(quantile_idx, len(sorted_scores) - 1)
        
        self.tau = sorted_scores[quantile_idx]
        
        # Store calibration results
        self.calibration_data = {
            'tau': self.tau,
            'total_trials': total_trials,
            'confidence_level': confidence_level,
            'all_scores': sorted_scores,
            'per_environment': detailed_results,
            'score_statistics': self.nonconformity_calc.analyze_score_distribution(all_scores)
        }
        
        self.calibration_completed = True
        
        # Log results
        self.logger.info(f"Global τ calibrated: {self.tau:.4f}m")
        self.logger.info(f"Based on {total_trials} trials across {len(test_environments)} environments")
        self.logger.info(f"Score statistics: mean={np.mean(all_scores):.4f}m, "
                        f"std={np.std(all_scores):.4f}m")
        
        # Save calibration results
        self.save_calibration_results()
        
        return self.tau
    
    def plan_with_standard_cp(self,
                             env_name: str,
                             test_id: int,
                             perceived_grid: Optional[np.ndarray] = None,
                             seed: int = 42) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path using Standard CP with global τ safety margin
        """
        if not self.calibration_completed:
            raise ValueError("Must calibrate τ first!")
        
        try:
            parser, env_config = self.load_environment(env_name)
            
            # Get test configuration
            test_config = env_config['tests'][test_id - 1]
            start = test_config['start']
            goal = test_config['goal']
            
            # Use perceived grid or true grid
            if perceived_grid is None:
                perceived_grid = parser.occupancy_grid
            
            # Apply safety margin by inflating obstacles
            inflated_grid = self._inflate_grid_by_tau(perceived_grid, self.tau)
            
            # Plan with inflated obstacles
            planner = RRTStarGrid(
                start=start,
                goal=goal,
                occupancy_grid=inflated_grid,
                origin=parser.origin,
                resolution=parser.resolution,
                robot_radius=self.config['robot']['radius'],  # Use original radius
                max_iter=self.config['planning']['rrt_star']['max_iterations'],
                seed=seed
            )
            
            # Plan with timeout protection
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Planning timeout")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.config['planning']['timeout']['planning_timeout']))
            
            try:
                path = planner.plan()
                signal.alarm(0)  # Cancel timeout
            except (TimeoutError, Exception) as e:
                signal.alarm(0)  # Cancel timeout
                self.logger.warning(f"Standard CP planning failed or timed out: {e}")
                path = None
            finally:
                signal.signal(signal.SIGALRM, old_handler)
            
            if path:
                # Apply quintic polynomial smoothing
                try:
                    trajectory_data = generate_realistic_trajectory(
                        path,
                        parser,
                        max_vel=self.config['planning']['smoothing']['max_velocity'],
                        max_acc=self.config['planning']['smoothing']['max_acceleration']
                    )
                    # Extract just the x,y coordinates from NavigationData
                    smooth_path = [(data.x, data.y) for data in trajectory_data]
                    return smooth_path
                except Exception as e:
                    self.logger.warning(f"Smoothing failed, using raw path: {e}")
                    return path
            
            return None
            
        except Exception as e:
            self.logger.error(f"Standard CP planning failed for {env_name}-{test_id}: {e}")
            return None
    
    def _inflate_grid_by_tau(self, grid: np.ndarray, tau: float) -> np.ndarray:
        """
        Inflate obstacles in occupancy grid by safety margin τ
        
        Args:
            grid: Original occupancy grid
            tau: Safety margin in meters
            
        Returns:
            Inflated occupancy grid
        """
        from scipy.ndimage import binary_dilation
        
        # Convert tau to pixels (grid resolution = 0.05m)
        tau_pixels = int(np.ceil(tau / 0.05))
        
        if tau_pixels <= 0:
            return grid.copy()
        
        # Create structuring element (circular)
        y, x = np.ogrid[-tau_pixels:tau_pixels+1, -tau_pixels:tau_pixels+1]
        mask = x*x + y*y <= tau_pixels*tau_pixels
        
        # Apply dilation to inflate obstacles
        inflated_grid = binary_dilation(grid, structure=mask).astype(int)
        
        self.logger.debug(f"Inflated obstacles by {tau:.3f}m ({tau_pixels} pixels)")
        
        return inflated_grid
    
    def evaluate_comparison(self, num_trials: int = None) -> Dict:
        """
        Comprehensive evaluation: Naive vs Standard CP
        
        Uses existing MRPB metrics for consistency
        """
        if not self.calibration_completed:
            raise ValueError("Must calibrate τ first!")
        
        if num_trials is None:
            if self.config['environments']['full_environments']['enabled']:
                num_trials = self.config['conformal_prediction']['calibration']['total_trials_full']
            else:
                num_trials = self.config['conformal_prediction']['calibration']['total_trials_test']
        
        self.logger.info(f"Starting evaluation with {num_trials} trials...")
        
        results = {
            'naive': {
                'success_count': 0,
                'collision_count': 0,
                'total_trials': 0,
                'planning_times': [],
                'path_lengths': [],
                'mrpb_metrics': []
            },
            'standard_cp': {
                'success_count': 0,
                'collision_count': 0,
                'total_trials': 0,
                'planning_times': [],
                'path_lengths': [],
                'mrpb_metrics': []
            }
        }
        
        test_environments = self.get_test_environments()
        noise_levels = self.config['noise_model']['noise_levels']
        
        trial_count = 0
        for trial in range(num_trials):
            # Sample environment and test case
            env_config = test_environments[trial % len(test_environments)]
            env_name = env_config['name']
            test_id = env_config['test_ids'][trial % len(env_config['test_ids'])]
            noise_level = noise_levels[trial % len(noise_levels)]
            
            # Load environment
            parser, mrpb_env_config = self.load_environment(env_name)
            true_grid = parser.occupancy_grid
            
            # Add perception noise
            trial_seed = self.config['conformal_prediction']['calibration']['random_seed_base'] + 10000 + trial
            perceived_grid = self.noise_model.add_realistic_noise(
                true_grid, noise_level=noise_level, seed=trial_seed
            )
            
            # Test both methods
            for method in ['naive', 'standard_cp']:
                start_time = time.time()
                
                if method == 'naive':
                    path = self.plan_naive_path(env_name, test_id, perceived_grid, seed=trial_seed + 1000)
                else:
                    path = self.plan_with_standard_cp(env_name, test_id, perceived_grid, seed=trial_seed + 2000)
                
                planning_time = time.time() - start_time
                results[method]['planning_times'].append(planning_time)
                results[method]['total_trials'] += 1
                
                if path:
                    results[method]['success_count'] += 1
                    
                    # Compute MRPB metrics using existing infrastructure
                    try:
                        metrics = self._compute_mrpb_metrics(path, env_name, true_grid)
                        results[method]['mrpb_metrics'].append(metrics)
                        results[method]['path_lengths'].append(metrics.get('path_length', len(path)))
                        
                        # Check collision on true environment
                        collision = self._check_collision_on_true_grid(path, true_grid)
                        if collision:
                            results[method]['collision_count'] += 1
                            
                    except Exception as e:
                        self.logger.warning(f"Metrics computation failed: {e}")
                        results[method]['path_lengths'].append(len(path))
            
            trial_count += 1
            if trial_count % 50 == 0:
                self.logger.info(f"Completed {trial_count}/{num_trials} evaluation trials...")
        
        # Compute summary statistics
        self._compute_evaluation_statistics(results)
        
        # Store results
        self.results['evaluation'] = results
        
        # Save results
        self.save_evaluation_results(results)
        
        return results
    
    def _compute_mrpb_metrics(self, path: List[Tuple[float, float]], env_name: str, true_grid: np.ndarray) -> Dict:
        """
        Compute MRPB metrics using existing infrastructure
        
        Reuses existing MRPBMetrics class for consistency
        """
        try:
            # Create NavigationData from path
            nav_data = []
            
            for i, (x, y) in enumerate(path):
                # Compute obstacle distance
                grid_x = int((x + true_grid.shape[1] * 0.05 / 2) / 0.05)
                grid_y = int((y + true_grid.shape[0] * 0.05 / 2) / 0.05)
                
                # Simple obstacle distance calculation
                obs_dist = self._compute_obstacle_distance(grid_x, grid_y, true_grid)
                
                nav_data.append(NavigationData(
                    x=x, y=y,
                    obs_dist=obs_dist,
                    t=i * 0.1,  # Assume 0.1s per waypoint
                    v=1.0,      # Assume constant velocity
                    a=0.0       # Assume no acceleration
                ))
            
            # Compute metrics
            metrics_calc = MRPBMetrics(nav_data)
            return metrics_calc.compute_all_metrics()
            
        except Exception as e:
            self.logger.warning(f"MRPB metrics computation failed: {e}")
            return {'path_length': len(path)}
    
    def _compute_obstacle_distance(self, grid_x: int, grid_y: int, grid: np.ndarray) -> float:
        """Simple obstacle distance computation"""
        height, width = grid.shape
        
        if not (0 <= grid_x < width and 0 <= grid_y < height):
            return 0.0
        
        min_dist = float('inf')
        search_radius = 20  # Search in 20-pixel radius
        
        for i in range(max(0, grid_y - search_radius), min(height, grid_y + search_radius + 1)):
            for j in range(max(0, grid_x - search_radius), min(width, grid_x + search_radius + 1)):
                if grid[i, j] == 100:  # MRPB occupied
                    dist = np.sqrt((grid_x - j)**2 + (grid_y - i)**2) * 0.05  # Convert to meters
                    min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != float('inf') else 10.0
    
    def _check_collision_on_true_grid(self, path: List[Tuple[float, float]], true_grid: np.ndarray) -> bool:
        """Check if path collides on true environment"""
        robot_radius = self.config['robot']['radius']
        
        for x, y in path:
            # Convert to grid coordinates
            grid_x = int((x + true_grid.shape[1] * 0.05 / 2) / 0.05)
            grid_y = int((y + true_grid.shape[0] * 0.05 / 2) / 0.05)
            
            # Check robot footprint
            robot_radius_pixels = int(np.ceil(robot_radius / 0.05))
            
            for di in range(-robot_radius_pixels, robot_radius_pixels + 1):
                for dj in range(-robot_radius_pixels, robot_radius_pixels + 1):
                    check_x, check_y = grid_x + dj, grid_y + di
                    
                    if (0 <= check_x < true_grid.shape[1] and 
                        0 <= check_y < true_grid.shape[0]):
                        
                        # Check if within robot radius
                        if di*di + dj*dj <= robot_radius_pixels*robot_radius_pixels:
                            if true_grid[check_y, check_x] == 100:  # MRPB occupied
                                return True  # Collision detected
        
        return False
    
    def _compute_evaluation_statistics(self, results: Dict):
        """Compute summary statistics for evaluation results"""
        for method in ['naive', 'standard_cp']:
            data = results[method]
            
            if data['total_trials'] > 0:
                data['success_rate'] = data['success_count'] / data['total_trials']
                data['collision_rate'] = data['collision_count'] / max(data['success_count'], 1)
                data['avg_planning_time'] = np.mean(data['planning_times'])
                
                if data['path_lengths']:
                    data['avg_path_length'] = np.mean(data['path_lengths'])
                    data['std_path_length'] = np.std(data['path_lengths'])
                else:
                    data['avg_path_length'] = 0.0
                    data['std_path_length'] = 0.0
    
    def save_calibration_results(self):
        """Save calibration results to both JSON and CSV files"""
        if not self.calibration_data:
            return
        
        timestamp = datetime.now().strftime(self.config['output']['timestamp_format'])
        
        # Create directories
        base_dir = self.config['output']['base_dir']
        data_dir = os.path.join(base_dir, self.config['output']['subdirs']['data'])
        results_dir = os.path.join(base_dir, self.config['output']['subdirs']['results'])
        calibration_dir = os.path.join(base_dir, self.config['output']['subdirs']['calibration'])
        
        # Save JSON data with proper type conversion
        results_file = os.path.join(data_dir, f"calibration_results_{timestamp}.json")
        save_data = self._convert_numpy_types(self.calibration_data.copy())
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        # Save CSV data for analysis
        self._save_calibration_csv(timestamp, calibration_dir, results_dir)
        
        self.logger.info(f"Calibration results saved to: {results_file}")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _save_calibration_csv(self, timestamp: str, calibration_dir: str, results_dir: str):
        """Save calibration data in CSV format for easy analysis"""
        import pandas as pd
        
        # 1. Save nonconformity scores CSV
        scores_data = []
        for env_name, env_data in self.calibration_data['per_environment'].items():
            for detail in env_data['details']:
                scores_data.append({
                    'timestamp': timestamp,
                    'environment': env_name,
                    'test_id': detail['test_id'],
                    'trial': detail['trial'],
                    'noise_level': detail['noise_level'],
                    'nonconformity_score': detail['score'],
                    'path_found': detail['path_found'],
                    'path_length': detail['path_length'],
                    'tau_global': self.calibration_data['tau']
                })
        
        scores_df = pd.DataFrame(scores_data)
        scores_file = os.path.join(results_dir, f"nonconformity_scores_{timestamp}.csv")
        scores_df.to_csv(scores_file, index=False)
        print(f"📊 Nonconformity scores saved: {scores_file}")
        
        # 2. Save calibration summary CSV
        summary_data = []
        for env_name, env_data in self.calibration_data['per_environment'].items():
            summary_data.append({
                'timestamp': timestamp,
                'environment': env_name,
                'mean_score': env_data['mean_score'],
                'std_score': env_data['std_score'], 
                'success_rate': env_data['success_rate'],
                'num_trials': len(env_data['scores']),
                'tau_global': self.calibration_data['tau']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(calibration_dir, f"calibration_summary_{timestamp}.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"📊 Calibration summary saved: {summary_file}")
        
        # 3. Save tau analysis CSV
        stats = self.calibration_data['score_statistics']
        tau_data = [{
            'timestamp': timestamp,
            'tau_value': self.calibration_data['tau'],
            'confidence_level': self.calibration_data['confidence_level'],
            'total_trials': self.calibration_data['total_trials'],
            'score_mean': stats['mean'],
            'score_std': stats['std'],
            'score_min': stats['min'],
            'score_max': stats['max'],
            'score_median': stats['median'],
            'score_90th_percentile': stats['percentiles']['90%'],
            'score_95th_percentile': stats['percentiles']['95%'],
            'zero_scores': stats['zero_scores'],
            'high_scores': stats['high_scores']
        }]
        
        tau_df = pd.DataFrame(tau_data)
        tau_file = os.path.join(calibration_dir, f"tau_analysis_{timestamp}.csv")
        tau_df.to_csv(tau_file, index=False)
        print(f"📊 Tau analysis saved: {tau_file}")
    
    def save_evaluation_results(self, results: Dict):
        """Save evaluation results to both JSON and CSV files"""
        timestamp = datetime.now().strftime(self.config['output']['timestamp_format'])
        
        # Create directories
        base_dir = self.config['output']['base_dir'] 
        data_dir = os.path.join(base_dir, self.config['output']['subdirs']['data'])
        results_dir = os.path.join(base_dir, self.config['output']['subdirs']['results'])
        evaluation_dir = os.path.join(base_dir, self.config['output']['subdirs']['evaluation'])
        
        # Save JSON data
        results_file = os.path.join(data_dir, f"evaluation_results_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV data for analysis
        self._save_evaluation_csv(timestamp, results, evaluation_dir, results_dir)
        
        self.logger.info(f"Evaluation results saved to: {results_file}")
    
    def _save_evaluation_csv(self, timestamp: str, results: Dict, evaluation_dir: str, results_dir: str):
        """Save evaluation results in CSV format"""
        import pandas as pd
        
        # 1. Save method comparison summary
        comparison_data = []
        for method, data in results.items():
            comparison_data.append({
                'timestamp': timestamp,
                'method': method,
                'total_trials': data.get('total_trials', 0),
                'success_count': data.get('success_count', 0),
                'success_rate': data.get('success_rate', 0),
                'collision_count': data.get('collision_count', 0),
                'collision_rate': data.get('collision_rate', 0),
                'avg_planning_time': data.get('avg_planning_time', 0),
                'avg_path_length': data.get('avg_path_length', 0),
                'std_path_length': data.get('std_path_length', 0),
                'tau_used': self.tau if method == 'standard_cp' else 0.0
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_file = os.path.join(evaluation_dir, f"method_comparison_{timestamp}.csv")
        comparison_df.to_csv(comparison_file, index=False)
        print(f"📊 Method comparison saved: {comparison_file}")
        
        # 2. Save detailed planning times for analysis
        planning_times_data = []
        for method, data in results.items():
            for i, time_val in enumerate(data.get('planning_times', [])):
                planning_times_data.append({
                    'timestamp': timestamp,
                    'method': method,
                    'trial': i,
                    'planning_time': time_val
                })
        
        if planning_times_data:
            times_df = pd.DataFrame(planning_times_data)
            times_file = os.path.join(results_dir, f"planning_times_{timestamp}.csv")
            times_df.to_csv(times_file, index=False)
            print(f"📊 Planning times saved: {times_file}")
        
        # 3. Save path lengths for analysis  
        path_lengths_data = []
        for method, data in results.items():
            for i, length_val in enumerate(data.get('path_lengths', [])):
                path_lengths_data.append({
                    'timestamp': timestamp,
                    'method': method,
                    'trial': i,
                    'path_length': length_val
                })
        
        if path_lengths_data:
            lengths_df = pd.DataFrame(path_lengths_data)
            lengths_file = os.path.join(results_dir, f"path_lengths_{timestamp}.csv")
            lengths_df.to_csv(lengths_file, index=False)
            print(f"📊 Path lengths saved: {lengths_file}")
            
        # 4. Save MRPB metrics if available
        mrpb_data = []
        for method, data in results.items():
            for i, metrics in enumerate(data.get('mrpb_metrics', [])):
                if metrics:
                    mrpb_entry = {
                        'timestamp': timestamp,
                        'method': method,
                        'trial': i,
                        'tau_used': self.tau if method == 'standard_cp' else 0.0
                    }
                    mrpb_entry.update(metrics)  # Add all MRPB metrics
                    mrpb_data.append(mrpb_entry)
        
        if mrpb_data:
            mrpb_df = pd.DataFrame(mrpb_data)
            mrpb_file = os.path.join(results_dir, f"mrpb_metrics_{timestamp}.csv")
            mrpb_df.to_csv(mrpb_file, index=False)
            print(f"📊 MRPB metrics saved: {mrpb_file}")
    
    def print_results_summary(self):
        """Print summary of results"""
        if not self.calibration_completed:
            print("No calibration completed yet.")
            return
        
        print("\n" + "="*80)
        print("STANDARD CP RESULTS SUMMARY")
        print("="*80)
        
        # Calibration summary
        print(f"\nCalibration:")
        print(f"  Global τ: {self.tau:.4f}m")
        print(f"  Confidence level: {self.config['conformal_prediction']['confidence_level']*100:.0f}%")
        if 'score_statistics' in self.calibration_data:
            stats = self.calibration_data['score_statistics']
            print(f"  Score statistics: mean={stats.get('mean', 0):.4f}m, "
                  f"std={stats.get('std', 0):.4f}m")
        
        # Evaluation summary
        if 'evaluation' in self.results and self.results['evaluation']:
            print(f"\nEvaluation:")
            eval_data = self.results['evaluation']
            
            for method in ['naive', 'standard_cp']:
                if method in eval_data:
                    data = eval_data[method]
                    print(f"  {method.replace('_', ' ').title()}:")
                    print(f"    Success rate: {data.get('success_rate', 0)*100:.1f}%")
                    print(f"    Collision rate: {data.get('collision_rate', 0)*100:.1f}%")
                    print(f"    Avg path length: {data.get('avg_path_length', 0):.2f}")
                    print(f"    Avg planning time: {data.get('avg_planning_time', 0)*1000:.1f}ms")


def main():
    """Main function for testing Standard CP implementation"""
    print("Standard CP Main - Testing Implementation")
    print("="*50)
    
    # Initialize planner
    planner = StandardCPPlanner()
    
    # Test with small number of trials for debugging
    print("\n1. Calibrating global τ...")
    tau = planner.calibrate_global_tau()
    print(f"Calibrated τ: {tau:.4f}m")
    
    # Test planning
    print("\n2. Testing path planning...")
    
    # Test naive planning
    naive_path = planner.plan_naive_path("office01add", 1, seed=42)
    print(f"Naive path found: {naive_path is not None}")
    if naive_path:
        print(f"Naive path length: {len(naive_path)} waypoints")
    
    # Test standard CP planning
    cp_path = planner.plan_with_standard_cp("office01add", 1, seed=42)
    print(f"Standard CP path found: {cp_path is not None}")
    if cp_path:
        print(f"Standard CP path length: {len(cp_path)} waypoints")
    
    # Small evaluation
    print("\n3. Running small evaluation...")
    results = planner.evaluate_comparison(num_trials=20)
    
    # Print summary
    planner.print_results_summary()
    
    print("\nStandard CP testing completed!")


if __name__ == "__main__":
    main()