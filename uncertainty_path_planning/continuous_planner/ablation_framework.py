#!/usr/bin/env python3
"""
Generalized Ablation Study Framework for ICRA 2025
Modular and extensible framework for various ablation studies
Can be used by all methods: Naive, Standard CP, Learnable CP
"""

import numpy as np
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any
from datetime import datetime
from abc import ABC, abstractmethod
import pandas as pd
from tqdm import tqdm

# Import shared modules
from noise_model import NoiseModel
from nonconformity_scorer import NonconformityScorer
from mrpb_map_parser import MRPBMapParser
from rrt_star_grid_planner import RRTStarGrid
from mrpb_metrics import MRPBMetrics


class AblationStudy(ABC):
    """
    Abstract base class for ablation studies
    Provides common functionality that can be extended for specific studies
    """
    
    def __init__(self, 
                 study_name: str,
                 config_paths: Dict[str, str] = None,
                 results_base_dir: str = "results/ablation_studies"):
        """
        Initialize ablation study framework
        
        Args:
            study_name: Name of the specific study
            config_paths: Dictionary of configuration file paths
            results_base_dir: Base directory for results
        """
        self.study_name = study_name
        
        # Default config paths
        if config_paths is None:
            config_paths = {
                'cp_config': 'config/standard_cp_config.yaml',
                'env_config': 'config/config_env.yaml'
            }
        
        # Load configurations
        self.configs = {}
        for name, path in config_paths.items():
            with open(path, 'r') as f:
                self.configs[name] = yaml.safe_load(f)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components (can be overridden)
        self.initialize_components()
        
        # Setup results directory
        self.results_dir = Path(results_base_dir) / study_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = {}
        self.metadata = {
            'study_name': study_name,
            'timestamp': datetime.now().isoformat(),
            'configs': config_paths
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'{self.study_name}_{datetime.now():%Y%m%d_%H%M%S}.log')
            ]
        )
        self.logger = logging.getLogger(self.study_name)
    
    def initialize_components(self):
        """Initialize common components - can be overridden"""
        self.noise_model = NoiseModel(list(self.configs.values())[0])
        self.nonconformity_scorer = NonconformityScorer(list(self.configs.values())[0])
        self.metrics_calculator = MRPBMetrics()
    
    @abstractmethod
    def run_experiment(self, **kwargs) -> Dict:
        """
        Run a single experiment - must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def analyze_results(self) -> Dict:
        """
        Analyze collected results - must be implemented by subclasses
        """
        pass
    
    def save_results(self, include_raw_data: bool = False):
        """Save results to multiple formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metadata
        metadata_path = self.results_dir / f"metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        # Save main results as JSON
        results_path = self.results_dir / f"results_{timestamp}.json"
        
        # Convert numpy types for JSON serialization
        json_results = self._convert_for_json(self.results)
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_path}")
        
        # Save summary as CSV if applicable
        if 'summary' in self.results:
            csv_path = self.results_dir / f"summary_{timestamp}.csv"
            df = pd.DataFrame(self.results['summary'])
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Summary saved to {csv_path}")
        
        return results_path
    
    def _convert_for_json(self, obj):
        """Recursively convert numpy types for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.number, np.bool_)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_for_json(item) for item in obj]
        else:
            return obj
    
    def load_environment(self, env_name: str, mrpb_path: str = "mrpb_dataset") -> MRPBMapParser:
        """Load and cache environment"""
        if not hasattr(self, '_env_cache'):
            self._env_cache = {}
        
        if env_name not in self._env_cache:
            self._env_cache[env_name] = MRPBMapParser(
                map_name=env_name,
                mrpb_path=mrpb_path
            )
        
        return self._env_cache[env_name]
    
    def plan_path(self,
                  start: List[float],
                  goal: List[float],
                  occupancy_grid: np.ndarray,
                  parser: MRPBMapParser,
                  robot_radius: float = 0.17,
                  max_iter: int = 10000,
                  early_termination: bool = True,
                  seed: int = None) -> Optional[List[Tuple[float, float]]]:
        """
        Common path planning function
        """
        planner = RRTStarGrid(
            start=start,
            goal=goal,
            occupancy_grid=occupancy_grid,
            origin=parser.origin,
            resolution=parser.resolution,
            robot_radius=robot_radius,
            max_iter=max_iter,
            early_termination=early_termination,
            seed=seed
        )
        
        try:
            path = planner.plan()
            return path
        except Exception as e:
            self.logger.debug(f"Planning failed: {e}")
            return None
    
    def calculate_tau(self, scores: List[float], confidence_level: float = 0.9) -> float:
        """Calculate tau (quantile) from scores"""
        if not scores:
            return 0.0
        
        sorted_scores = sorted(scores)
        quantile_idx = int(np.ceil((len(sorted_scores) + 1) * confidence_level)) - 1
        quantile_idx = min(quantile_idx, len(sorted_scores) - 1)
        
        return sorted_scores[quantile_idx]
    
    def calculate_statistics(self, values: List[float]) -> Dict:
        """Calculate comprehensive statistics"""
        if not values:
            return {
                'count': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'percentiles': {}
            }
        
        arr = np.array(values)
        return {
            'count': len(values),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'median': float(np.median(arr)),
            'percentiles': {
                '25%': float(np.percentile(arr, 25)),
                '50%': float(np.percentile(arr, 50)),
                '75%': float(np.percentile(arr, 75)),
                '90%': float(np.percentile(arr, 90)),
                '95%': float(np.percentile(arr, 95)),
                '99%': float(np.percentile(arr, 99))
            }
        }


class NoiseAblationStudy(AblationStudy):
    """
    Specific implementation for noise type ablation
    """
    
    def __init__(self, **kwargs):
        super().__init__(study_name="noise_ablation", **kwargs)
        
        # Define noise configurations
        self.noise_configs = {
            "measurement_only": {
                "noise_types": ["measurement_noise"],
                "description": "LiDAR/Camera measurement noise only"
            },
            "false_negatives_only": {
                "noise_types": ["false_negatives"],
                "description": "Missing obstacle detections only"
            },
            "localization_only": {
                "noise_types": ["localization_drift"],
                "description": "Robot position uncertainty only"
            },
            "all_combined": {
                "noise_types": ["measurement_noise", "false_negatives", "localization_drift"],
                "description": "All noise types combined"
            }
        }
    
    def run_experiment(self, 
                      noise_config: Dict,
                      num_trials: int = 100,
                      environments: List[str] = None,
                      **kwargs) -> Dict:
        """
        Run noise ablation experiment
        """
        self.logger.info(f"Running experiment: {noise_config['description']}")
        
        # Save original noise configuration
        original_noise_types = self.noise_model.noise_config['noise_types']
        self.noise_model.noise_config['noise_types'] = noise_config['noise_types']
        
        # Default environments
        if environments is None:
            environments = ['office01add', 'room02']
        
        # Collect scores
        all_scores = []
        trial_details = []
        
        # Run trials
        for env_name in environments:
            parser = self.load_environment(env_name)
            # ... (implementation details)
        
        # Restore original configuration
        self.noise_model.noise_config['noise_types'] = original_noise_types
        
        # Calculate statistics
        tau = self.calculate_tau(all_scores)
        stats = self.calculate_statistics(all_scores)
        
        return {
            'scores': all_scores,
            'tau': tau,
            'statistics': stats,
            'trial_details': trial_details
        }
    
    def analyze_results(self) -> Dict:
        """Analyze noise ablation results"""
        analysis = {}
        
        # Find most impactful noise type
        if self.results:
            max_tau_config = max(self.results.items(), 
                               key=lambda x: x[1].get('tau', 0))
            analysis['most_impactful'] = max_tau_config[0]
            analysis['highest_tau'] = max_tau_config[1]['tau']
        
        return analysis
    
    def run_full_study(self, num_trials: int = 100):
        """Run complete noise ablation study"""
        self.logger.info("Starting Noise Type Ablation Study")
        
        for noise_name, noise_config in self.noise_configs.items():
            self.results[noise_name] = self.run_experiment(
                noise_config=noise_config,
                num_trials=num_trials
            )
        
        # Analyze and save
        self.results['analysis'] = self.analyze_results()
        self.save_results()
        
        return self.results


class MonteCarloConvergenceStudy(AblationStudy):
    """
    Monte Carlo convergence analysis
    """
    
    def __init__(self, **kwargs):
        super().__init__(study_name="monte_carlo_convergence", **kwargs)
        self.sample_sizes = [10, 50, 100, 200, 500, 1000, 2000]
    
    def run_experiment(self, sample_size: int, **kwargs) -> Dict:
        """Run experiment for specific sample size"""
        self.logger.info(f"Running Monte Carlo with {sample_size} samples")
        
        # Implementation would go here
        # This is a placeholder structure
        scores = []
        for i in range(sample_size):
            # Generate trial
            score = np.random.random() * 0.3  # Placeholder
            scores.append(score)
        
        tau = self.calculate_tau(scores)
        stats = self.calculate_statistics(scores)
        
        return {
            'sample_size': sample_size,
            'tau': tau,
            'statistics': stats,
            'confidence_interval': self._calculate_confidence_interval(scores)
        }
    
    def _calculate_confidence_interval(self, scores: List[float], alpha: float = 0.05):
        """Calculate confidence interval using bootstrap"""
        if not scores:
            return [0.0, 0.0]
        
        # Bootstrap resampling
        n_bootstrap = 1000
        bootstrap_taus = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_taus.append(self.calculate_tau(sample.tolist()))
        
        lower = np.percentile(bootstrap_taus, alpha/2 * 100)
        upper = np.percentile(bootstrap_taus, (1 - alpha/2) * 100)
        
        return [float(lower), float(upper)]
    
    def analyze_results(self) -> Dict:
        """Analyze convergence"""
        if not self.results:
            return {}
        
        # Check convergence
        taus = [r['tau'] for r in self.results.values()]
        converged_idx = None
        
        for i in range(1, len(taus)):
            if abs(taus[i] - taus[i-1]) < 0.001:  # Convergence threshold
                converged_idx = i
                break
        
        return {
            'converged': converged_idx is not None,
            'convergence_sample_size': self.sample_sizes[converged_idx] if converged_idx else None,
            'final_tau': taus[-1] if taus else None
        }
    
    def run_full_study(self):
        """Run complete convergence study"""
        self.logger.info("Starting Monte Carlo Convergence Study")
        
        for sample_size in self.sample_sizes:
            self.results[f'n_{sample_size}'] = self.run_experiment(sample_size)
        
        self.results['analysis'] = self.analyze_results()
        self.save_results()
        
        return self.results


# Factory function for creating studies
def create_ablation_study(study_type: str, **kwargs) -> AblationStudy:
    """
    Factory function to create specific ablation studies
    
    Args:
        study_type: Type of study ('noise', 'monte_carlo', etc.)
        kwargs: Additional arguments for the study
    
    Returns:
        AblationStudy instance
    """
    studies = {
        'noise': NoiseAblationStudy,
        'monte_carlo': MonteCarloConvergenceStudy,
        # Add more study types here as needed
    }
    
    if study_type not in studies:
        raise ValueError(f"Unknown study type: {study_type}")
    
    return studies[study_type](**kwargs)


if __name__ == "__main__":
    # Example usage
    print("Ablation Framework Module - Import this to use in your studies")
    print("Available studies:", ['noise', 'monte_carlo'])
    
    # Quick test
    study = create_ablation_study('noise')
    print(f"Created study: {study.study_name}")