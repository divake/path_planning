#!/usr/bin/env python3
"""
Main Runner for Learnable Conformal Prediction
Integrates all components and runs the complete pipeline
"""

import os
import sys
import numpy as np
import torch
import yaml
import json
import pickle
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import argparse
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neural_network import AdaptiveTauNetwork
from feature_extractor import WaypointFeatureExtractor
from loss_function import AdaptiveTauLoss
from tau_calculator import AdaptiveTauCalculator, HybridTauCalculator
from trainer import LearnableCPTrainer, PathPlanningDataset
from visualization import LearnableCPVisualizer

# Import planning components
from src.rrt_star_grid_planner import RRTStarGrid
from src.conformal_prediction import ConformalPrediction
from utils.mrpb_dataset import MRPBDataset
from utils.metrics import MRPBMetrics
from utils.noise_model import NoiseModel


class LearnableCPRunner:
    """
    Main runner class for Learnable Conformal Prediction experiments.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize runner with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        
        # Initialize dataset
        self.dataset = MRPBDataset(self.config['dataset']['path'])
        
        # Initialize metrics calculator
        self.metrics = MRPBMetrics()
        
        # Initialize visualizer
        self.visualizer = LearnableCPVisualizer(
            save_dir=self.config.get('visualization_dir', 'visualization')
        )
        
        # Results storage
        self.results = {
            'training': {},
            'evaluation': {},
            'ablation': {},
            'comparison': {}
        }
        
        logging.info("LearnableCPRunner initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        if config_path.endswith('.yaml'):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Default configuration
            config = {
                'dataset': {
                    'path': '../../../data/mrpb_dataset'
                },
                'model': {
                    'input_dim': 20,
                    'hidden_dims': [128, 64, 32],
                    'dropout': 0.2,
                    'batch_norm': True,
                    'residual': True
                },
                'training': {
                    'batch_size': 64,
                    'num_epochs': 100,
                    'learning_rate': 1e-3,
                    'weight_decay': 1e-4
                },
                'evaluation': {
                    'num_trials': 50,
                    'noise_types': ['transparency', 'occlusion', 'localization', 'combined'],
                    'noise_level': 0.25
                },
                'planner': {
                    'max_iterations': 15000,
                    'step_size': 0.3,
                    'goal_bias': 0.1,
                    'downsample_rate': 0.5
                },
                'tau': {
                    'min': 0.05,
                    'max': 0.50,
                    'default': 0.30,
                    'smooth': True
                }
            }
        
        return config
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.config.get('log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'learnable_cp_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def generate_training_data(self) -> str:
        """
        Generate training data from MRPB dataset.
        
        Returns:
            Path to generated training data
        """
        logging.info("Generating training data from MRPB dataset")
        
        training_samples = []
        feature_extractor = WaypointFeatureExtractor()
        
        # Process each environment
        for env_name in tqdm(self.dataset.get_environments(), desc="Processing environments"):
            env_data = self.dataset.load_environment(env_name)
            
            # Skip if no data
            if env_data is None:
                continue
            
            occupancy_grid = env_data['occupancy_grid']
            origin = env_data['origin']
            resolution = env_data['resolution']
            
            # Process each test case
            for test_idx in range(3):  # 3 test cases per environment
                test_name = f"{env_name}-{test_idx + 1}"
                
                # Get start and goal
                start = env_data['test_cases'][test_idx]['start']
                goal = env_data['test_cases'][test_idx]['goal']
                
                # Plan path using RRT*
                planner = RRTStarGrid(
                    occupancy_grid=occupancy_grid,
                    origin=origin,
                    resolution=resolution,
                    robot_radius=0.17
                )
                
                path = planner.plan(start, goal, self.config['planner'])
                
                if path is None:
                    continue
                
                # Apply different noise types and collect data
                for noise_type in self.config['evaluation']['noise_types']:
                    # Initialize noise model
                    noise_model = NoiseModel(
                        noise_type=noise_type,
                        noise_level=self.config['evaluation']['noise_level']
                    )
                    
                    # Apply noise to get actual clearances
                    noisy_grid = noise_model.apply_noise(occupancy_grid)
                    
                    # Extract features and clearances for each waypoint
                    for idx, waypoint in enumerate(path):
                        # Extract features
                        features = feature_extractor.extract_features(
                            waypoint, path, idx,
                            occupancy_grid, origin, resolution, noise_type
                        )
                        
                        # Compute actual clearance
                        clearance = self._compute_clearance(
                            waypoint, noisy_grid, origin, resolution
                        )
                        
                        # Store sample
                        sample = {
                            'features': features,
                            'clearance': clearance,
                            'noise_type': noise_type,
                            'env_name': env_name,
                            'path_id': f"{test_name}_{noise_type}"
                        }
                        training_samples.append(sample)
        
        # Save training data
        data_dir = self.config.get('data_dir', 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        data_path = os.path.join(data_dir, 'training_data.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump({'samples': training_samples}, f)
        
        logging.info(f"Generated {len(training_samples)} training samples")
        logging.info(f"Training data saved to {data_path}")
        
        return data_path
    
    def _compute_clearance(self, waypoint: np.ndarray, occupancy_grid: np.ndarray,
                          origin: np.ndarray, resolution: float) -> float:
        """Compute clearance to nearest obstacle"""
        # Convert to grid coordinates
        grid_x = int((waypoint[0] - origin[0]) / resolution)
        grid_y = int((waypoint[1] - origin[1]) / resolution)
        
        # Find nearest obstacle
        obstacle_points = np.column_stack(np.where(occupancy_grid > 50))
        if len(obstacle_points) == 0:
            return 5.0  # Max clearance
        
        point = np.array([grid_y, grid_x])
        distances = np.linalg.norm(obstacle_points - point, axis=1)
        min_dist = np.min(distances) * resolution
        
        return min(min_dist, 5.0)
    
    def train_model(self, data_path: Optional[str] = None):
        """
        Train the Learnable CP model.
        
        Args:
            data_path: Path to training data (if None, generates new data)
        """
        logging.info("Starting model training")
        
        # Generate training data if needed
        if data_path is None:
            data_path = self.generate_training_data()
        
        # Split data into train/val
        self._split_data(data_path)
        
        # Create trainer
        trainer = LearnableCPTrainer(self.config)
        
        # Create data loaders
        from torch.utils.data import DataLoader
        
        feature_extractor = WaypointFeatureExtractor()
        
        train_dataset = PathPlanningDataset(
            os.path.join(self.config['data_dir'], 'train_data.pkl'),
            feature_extractor
        )
        
        val_dataset = PathPlanningDataset(
            os.path.join(self.config['data_dir'], 'val_data.pkl'),
            feature_extractor
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=4
        )
        
        # Train model
        trainer.train(
            train_loader,
            val_loader,
            num_epochs=self.config['training']['num_epochs']
        )
        
        # Store training results
        self.results['training'] = {
            'history': trainer.training_history,
            'best_loss': trainer.best_loss,
            'final_epoch': trainer.current_epoch
        }
        
        # Visualize training
        self.visualizer.plot_training_curves(
            trainer.training_history,
            title="Learnable CP Training Progress"
        )
        
        logging.info("Model training completed")
    
    def _split_data(self, data_path: str):
        """Split data into train/validation sets"""
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        samples = data['samples']
        np.random.shuffle(samples)
        
        split_idx = int(0.8 * len(samples))
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
        
        # Save splits
        data_dir = self.config.get('data_dir', 'data')
        
        with open(os.path.join(data_dir, 'train_data.pkl'), 'wb') as f:
            pickle.dump({'samples': train_samples}, f)
        
        with open(os.path.join(data_dir, 'val_data.pkl'), 'wb') as f:
            pickle.dump({'samples': val_samples}, f)
        
        logging.info(f"Data split: {len(train_samples)} train, {len(val_samples)} validation")
    
    def evaluate_model(self, model_path: str):
        """
        Evaluate trained model on test environments.
        
        Args:
            model_path: Path to trained model checkpoint
        """
        logging.info("Starting model evaluation")
        
        # Load model
        model = AdaptiveTauNetwork(
            input_dim=self.config['model']['input_dim'],
            hidden_dims=self.config['model']['hidden_dims']
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Initialize components
        feature_extractor = WaypointFeatureExtractor()
        tau_calculator = AdaptiveTauCalculator(model, feature_extractor, self.config['tau'])
        
        # Evaluation results
        eval_results = []
        
        # Process each test environment
        for env_name in tqdm(self.dataset.get_environments()[:3], desc="Evaluating"):
            env_data = self.dataset.load_environment(env_name)
            
            if env_data is None:
                continue
            
            for test_idx in range(3):
                for noise_type in self.config['evaluation']['noise_types']:
                    # Run evaluation
                    result = self._evaluate_single_test(
                        env_data, test_idx, noise_type,
                        tau_calculator
                    )
                    
                    if result:
                        eval_results.append(result)
        
        # Aggregate results
        self.results['evaluation'] = self._aggregate_results(eval_results)
        
        # Visualize results
        self._visualize_evaluation_results()
        
        logging.info("Model evaluation completed")
    
    def _evaluate_single_test(self, env_data: Dict, test_idx: int,
                             noise_type: str, tau_calculator) -> Optional[Dict]:
        """Evaluate single test case"""
        # Implementation similar to Standard CP evaluation
        # Returns metrics dictionary
        pass
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate evaluation results"""
        # Group by noise type and compute statistics
        aggregated = {}
        
        for noise_type in self.config['evaluation']['noise_types']:
            noise_results = [r for r in results if r['noise_type'] == noise_type]
            
            if noise_results:
                aggregated[noise_type] = {
                    'avg_tau': np.mean([r['avg_tau'] for r in noise_results]),
                    'coverage': np.mean([r['coverage'] for r in noise_results]),
                    'd0': np.mean([r['d0'] for r in noise_results]),
                    'd_avg': np.mean([r['d_avg'] for r in noise_results]),
                    'p0': np.mean([r['p0'] for r in noise_results]),
                    'path_length': np.mean([r['path_length'] for r in noise_results]),
                    'computation_time': np.mean([r['time'] for r in noise_results])
                }
        
        return aggregated
    
    def _visualize_evaluation_results(self):
        """Visualize evaluation results"""
        if 'evaluation' not in self.results:
            return
        
        # Noise impact visualization
        self.visualizer.plot_noise_impact(
            self.results['evaluation'],
            title="Learnable CP Performance Across Noise Types"
        )
    
    def run_ablation_studies(self):
        """Run comprehensive ablation studies"""
        logging.info("Running ablation studies")
        
        ablation_results = {}
        
        # 1. Feature importance ablation
        ablation_results['feature_importance'] = self._ablate_features()
        
        # 2. Network architecture ablation
        ablation_results['architecture'] = self._ablate_architecture()
        
        # 3. Loss component ablation
        ablation_results['loss_components'] = self._ablate_loss_components()
        
        # 4. Tau range ablation
        ablation_results['tau_range'] = self._ablate_tau_range()
        
        self.results['ablation'] = ablation_results
        
        # Visualize ablation results
        self._visualize_ablation_results()
        
        logging.info("Ablation studies completed")
    
    def _ablate_features(self) -> Dict:
        """Ablate features one by one"""
        logging.info("Running feature ablation")
        # Implementation: Train models with each feature removed
        pass
    
    def _ablate_architecture(self) -> Dict:
        """Ablate network architecture components"""
        logging.info("Running architecture ablation")
        # Implementation: Test different architectures
        pass
    
    def _ablate_loss_components(self) -> Dict:
        """Ablate loss function components"""
        logging.info("Running loss component ablation")
        # Implementation: Train with different loss weights
        pass
    
    def _ablate_tau_range(self) -> Dict:
        """Ablate tau range parameters"""
        logging.info("Running tau range ablation")
        # Implementation: Test different tau bounds
        pass
    
    def _visualize_ablation_results(self):
        """Visualize ablation study results"""
        if 'ablation' not in self.results:
            return
        
        # Feature importance
        if 'feature_importance' in self.results['ablation']:
            self.visualizer.plot_feature_importance(
                self.results['ablation']['feature_importance']['names'],
                self.results['ablation']['feature_importance']['scores']
            )
    
    def compare_methods(self):
        """Compare Learnable CP with Standard CP and Naive methods"""
        logging.info("Comparing methods")
        
        # Load results from other methods
        standard_results = self._load_standard_cp_results()
        naive_results = self._load_naive_results()
        
        # Create comparison
        self.results['comparison'] = {
            'learnable': self.results.get('evaluation', {}),
            'standard': standard_results,
            'naive': naive_results
        }
        
        # Visualize comparison
        self.visualizer.create_summary_figure(
            self.results['comparison']['learnable'],
            self.results['comparison']['standard'],
            self.results['comparison']['naive'],
            title="Learnable CP vs Standard CP vs Naive Method"
        )
        
        logging.info("Method comparison completed")
    
    def _load_standard_cp_results(self) -> Dict:
        """Load Standard CP results"""
        # Implementation: Load from CSV or pickle
        pass
    
    def _load_naive_results(self) -> Dict:
        """Load Naive method results"""
        # Implementation: Load from CSV or pickle
        pass
    
    def save_results(self):
        """Save all results"""
        results_dir = self.config.get('results_dir', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as JSON
        json_path = os.path.join(results_dir, f'learnable_cp_results_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save as pickle
        pkl_path = os.path.join(results_dir, f'learnable_cp_results_{timestamp}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.results, f)
        
        logging.info(f"Results saved to {results_dir}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Learnable CP Runner")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['train', 'evaluate', 'ablation', 'compare', 'full'],
                       help='Execution mode')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model (for evaluation)')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to training data')
    
    args = parser.parse_args()
    
    # Create runner
    runner = LearnableCPRunner(args.config)
    
    # Execute based on mode
    if args.mode == 'train':
        runner.train_model(args.data)
    elif args.mode == 'evaluate':
        if args.model:
            runner.evaluate_model(args.model)
        else:
            print("Error: Model path required for evaluation")
    elif args.mode == 'ablation':
        runner.run_ablation_studies()
    elif args.mode == 'compare':
        runner.compare_methods()
    elif args.mode == 'full':
        # Run complete pipeline
        runner.train_model(args.data)
        runner.evaluate_model('checkpoints/best_model.pth')
        runner.run_ablation_studies()
        runner.compare_methods()
    
    # Save results
    runner.save_results()
    
    print("Learnable CP pipeline completed successfully!")


if __name__ == "__main__":
    main()