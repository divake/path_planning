#!/usr/bin/env python3
"""
Learnable CP Trainer for Path Planning
Trains the scoring function to predict adaptive safety margins.
Generalizable to work with ANY path planning algorithm (RRT*, A*, PRM, etc.)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import yaml
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from learn_cp_scoring_function import LearnableCPScoringFunction
from standard_cp_nonconformity import StandardCPNonconformity
from mrpb_map_parser import MRPBMapParser


class PathPlanningDataset(Dataset):
    """
    Dataset for training learnable CP.
    Collects path planning experiences from any planner.
    """
    
    def __init__(self, config: Dict, split: str = 'train'):
        """
        Initialize dataset.
        
        Args:
            config: Configuration dictionary
            split: 'train', 'val', or 'test'
        """
        self.config = config
        self.split = split
        self.data = []
        
        # Load environments from config_env.yaml
        with open('config_env.yaml', 'r') as f:
            env_config = yaml.safe_load(f)
        
        self.environments = env_config['environments']
        
        # Split environments for train/val/test
        env_names = list(self.environments.keys())
        np.random.seed(42)  # Fixed seed for reproducibility
        np.random.shuffle(env_names)
        
        n_envs = len(env_names)
        n_train = int(0.7 * n_envs)
        n_val = int(0.15 * n_envs)
        
        if split == 'train':
            self.env_names = env_names[:n_train]
        elif split == 'val':
            self.env_names = env_names[n_train:n_train+n_val]
        else:  # test
            self.env_names = env_names[n_train+n_val:]
        
        # Generate dataset
        self._generate_data()
    
    def _generate_data(self):
        """Generate training data from environments"""
        logging.info(f"Generating {self.split} data from {len(self.env_names)} environments")
        
        for env_name in self.env_names:
            try:
                # Parse environment
                parser = MRPBMapParser(env_name)
                obstacles = np.array(parser.extract_obstacles())
                env_info = parser.get_environment_info()
                bounds = [
                    env_info['origin'][0],
                    env_info['origin'][0] + env_info['width_meters'],
                    env_info['origin'][1],
                    env_info['origin'][1] + env_info['height_meters']
                ]
                
                # Get test configurations
                env_config = self.environments[env_name]
                if 'tests' not in env_config:
                    continue
                
                for test in env_config['tests']:
                    start = np.array(test['start'])
                    goal = np.array(test['goal'])
                    
                    # Generate path waypoints (can be from any planner)
                    # For now, sample points along straight line + random perturbations
                    num_waypoints = 20
                    t = np.linspace(0, 1, num_waypoints)
                    base_path = start[np.newaxis, :] + t[:, np.newaxis] * (goal - start)[np.newaxis, :]
                    
                    # Add some perturbations to simulate different path types
                    perturbations = np.random.randn(num_waypoints, 2) * 0.5
                    perturbations[0] = 0  # Keep start fixed
                    perturbations[-1] = 0  # Keep goal fixed
                    path_points = base_path + perturbations
                    
                    # For each point, compute ground truth safety requirement
                    for point in path_points:
                        # Compute actual clearance
                        min_clearance = self._compute_min_clearance(point, obstacles)
                        
                        # Label: Does this point need high tau?
                        # High tau needed if: close to obstacles, in narrow passage, etc.
                        needs_high_tau = (
                            min_clearance < 0.5 or  # Very close to obstacle
                            self._in_narrow_passage(point, obstacles) or
                            self._near_corner(point, obstacles)
                        )
                        
                        # Store data point
                        self.data.append({
                            'state': point,
                            'obstacles': obstacles,
                            'goal': goal,
                            'bounds': bounds,
                            'min_clearance': min_clearance,
                            'needs_high_tau': needs_high_tau,
                            'env_name': env_name,
                            'env_difficulty': env_config.get('difficulty', 'medium')
                        })
                        
            except Exception as e:
                logging.warning(f"Failed to process environment {env_name}: {e}")
                continue
        
        logging.info(f"Generated {len(self.data)} data points for {self.split}")
    
    def _compute_min_clearance(self, point: np.ndarray, obstacles: np.ndarray) -> float:
        """Compute minimum clearance to obstacles"""
        if len(obstacles) == 0:
            return 10.0
        
        min_dist = float('inf')
        for obs in obstacles:
            dist = self._point_to_rectangle_distance(point, obs)
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _point_to_rectangle_distance(self, point: np.ndarray, rect: np.ndarray) -> float:
        """Distance from point to rectangle"""
        x, y = point
        x1, y1, x2, y2 = rect
        
        if x1 <= x <= x2 and y1 <= y <= y2:
            return 0.0
        
        dx = max(x1 - x, 0, x - x2)
        dy = max(y1 - y, 0, y - y2)
        return np.sqrt(dx*dx + dy*dy)
    
    def _in_narrow_passage(self, point: np.ndarray, obstacles: np.ndarray) -> bool:
        """Check if point is in narrow passage"""
        # Simple heuristic: count nearby obstacles
        nearby_count = 0
        for obs in obstacles:
            if self._point_to_rectangle_distance(point, obs) < 2.0:
                nearby_count += 1
        return nearby_count >= 2
    
    def _near_corner(self, point: np.ndarray, obstacles: np.ndarray) -> bool:
        """Check if near corner"""
        close_count = 0
        for obs in obstacles:
            if self._point_to_rectangle_distance(point, obs) < 1.0:
                close_count += 1
        return close_count >= 2
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'state': torch.tensor(item['state'], dtype=torch.float32),
            'obstacles': item['obstacles'],
            'goal': torch.tensor(item['goal'], dtype=torch.float32),
            'bounds': item['bounds'],
            'min_clearance': torch.tensor(item['min_clearance'], dtype=torch.float32),
            'needs_high_tau': torch.tensor(float(item['needs_high_tau']), dtype=torch.float32),
            'env_difficulty': item['env_difficulty']
        }


class LearnableCPTrainer:
    """
    Trainer for Learnable CP scoring function.
    Generalizable to work with any path planning algorithm.
    """
    
    def __init__(self, config_path: str = "learn_cp_config.yaml"):
        """Initialize trainer with configuration"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = LearnableCPScoringFunction(self.config).to(self.device)
        
        # Setup results directory
        self.results_dir = Path("results/learn_cp")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup model checkpoint directory
        self.checkpoint_dir = self.results_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize datasets
        self.train_dataset = PathPlanningDataset(self.config, split='train')
        self.val_dataset = PathPlanningDataset(self.config, split='val')
        self.test_dataset = PathPlanningDataset(self.config, split='test')
        
        # Create dataloaders
        batch_size = self.config['training']['batch_size']
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=self._collate_fn
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Initialize tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Loss weights from config
        self.lambda_coverage = self.config['training']['loss_weights']['coverage']
        self.lambda_efficiency = self.config['training']['loss_weights']['efficiency']
        self.lambda_smoothness = self.config['training']['loss_weights'].get('smoothness', 0.1)
        
    def _collate_fn(self, batch):
        """Custom collate function for batching"""
        states = torch.stack([item['state'] for item in batch])
        goals = torch.stack([item['goal'] for item in batch])
        obstacles_list = [item['obstacles'] for item in batch]
        bounds_list = [item['bounds'] for item in batch]
        min_clearances = torch.stack([item['min_clearance'] for item in batch])
        needs_high_tau = torch.stack([item['needs_high_tau'] for item in batch])
        
        return {
            'states': states,
            'goals': goals,
            'obstacles_list': obstacles_list,
            'bounds_list': bounds_list,
            'min_clearances': min_clearances,
            'needs_high_tau': needs_high_tau
        }
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        opt_config = self.config['optimizer']
        
        if opt_config['type'] == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0.01)
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr']
            )
        
        # Learning rate scheduler
        if opt_config.get('scheduler', {}).get('type') == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )
        else:
            self.scheduler = None
    
    def compute_loss(self, predicted_tau: torch.Tensor, min_clearances: torch.Tensor, 
                    needs_high_tau: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute training loss.
        
        Loss components:
        1. Coverage loss: Ensure safety (tau >= required)
        2. Efficiency loss: Minimize unnecessary inflation
        3. Smoothness loss: Encourage smooth tau transitions
        """
        batch_size = predicted_tau.shape[0]
        
        # Coverage loss: Penalize when tau < required safety margin
        # Use min_clearance as proxy for required tau
        coverage_loss = torch.mean(
            needs_high_tau * torch.relu(0.2 - predicted_tau) +  # High risk areas need tau >= 0.2
            (1 - needs_high_tau) * torch.relu(predicted_tau - min_clearances)  # Don't over-inflate in safe areas
        )
        
        # Efficiency loss: Minimize tau when safe
        efficiency_loss = torch.mean(
            (1 - needs_high_tau) * predicted_tau  # Penalize large tau in safe areas
        )
        
        # Smoothness loss: Encourage smooth transitions
        if batch_size > 1:
            tau_diff = torch.diff(predicted_tau)
            smoothness_loss = torch.mean(tau_diff ** 2)
        else:
            smoothness_loss = torch.tensor(0.0)
        
        # Total loss
        total_loss = (
            self.lambda_coverage * coverage_loss +
            self.lambda_efficiency * efficiency_loss +
            self.lambda_smoothness * smoothness_loss
        )
        
        # Add L2 regularization from model
        if hasattr(self.model, 'l2_reg'):
            total_loss += self.model.l2_reg
        
        # Return losses for logging
        loss_dict = {
            'total': total_loss.item(),
            'coverage': coverage_loss.item(),
            'efficiency': efficiency_loss.item(),
            'smoothness': smoothness_loss.item()
        }
        
        return total_loss, loss_dict
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {
            'total': [], 'coverage': [], 
            'efficiency': [], 'smoothness': []
        }
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch}') as pbar:
            for batch in pbar:
                # Move to device
                states = batch['states'].to(self.device)
                goals = batch['goals'].to(self.device)
                min_clearances = batch['min_clearances'].to(self.device)
                needs_high_tau = batch['needs_high_tau'].to(self.device)
                
                # Forward pass
                predicted_tau = self.model(
                    states, 
                    batch['obstacles_list'],
                    goals,
                    batch['bounds_list']
                )
                
                # Compute loss
                loss, loss_dict = self.compute_loss(
                    predicted_tau, min_clearances, needs_high_tau
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Track losses
                for key, val in loss_dict.items():
                    epoch_losses[key].append(val)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'tau_mean': f"{predicted_tau.mean().item():.3f}"
                })
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        return avg_losses
    
    def validate(self) -> Dict:
        """Validate model"""
        self.model.eval()
        val_losses = {
            'total': [], 'coverage': [],
            'efficiency': [], 'smoothness': []
        }
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                states = batch['states'].to(self.device)
                goals = batch['goals'].to(self.device)
                min_clearances = batch['min_clearances'].to(self.device)
                needs_high_tau = batch['needs_high_tau'].to(self.device)
                
                # Forward pass
                predicted_tau = self.model(
                    states,
                    batch['obstacles_list'],
                    goals,
                    batch['bounds_list']
                )
                
                # Compute loss
                loss, loss_dict = self.compute_loss(
                    predicted_tau, min_clearances, needs_high_tau
                )
                
                # Track losses
                for key, val in loss_dict.items():
                    val_losses[key].append(val)
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in val_losses.items()}
        return avg_losses
    
    def train(self):
        """Main training loop"""
        num_epochs = self.config['training']['epochs']
        
        logging.info(f"Starting training for {num_epochs} epochs")
        logging.info(f"Train samples: {len(self.train_dataset)}")
        logging.info(f"Val samples: {len(self.val_dataset)}")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_losses = self.train_epoch(epoch)
            self.train_losses.append(train_losses)
            
            # Validate
            val_losses = self.validate()
            self.val_losses.append(val_losses)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Logging
            logging.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_losses['total']:.4f}, "
                f"Val Loss: {val_losses['total']:.4f}"
            )
            
            # Save best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
            
            # Regular checkpoint
            if epoch % self.config['training'].get('checkpoint_interval', 10) == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        logging.info(f"Training complete! Best epoch: {self.best_epoch}")
        
        # Save training history
        self.save_training_history()
        
        # Generate plots
        self.plot_training_curves()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save checkpoint
        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
            logging.info(f"Saving best model (epoch {epoch})")
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str = None):
        """Load model checkpoint"""
        if path is None:
            path = self.checkpoint_dir / 'best_model.pth'
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logging.info(f"Loaded checkpoint from {path}")
        return checkpoint['epoch']
    
    def save_training_history(self):
        """Save training history to JSON"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        path = self.results_dir / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logging.info(f"Saved training history to {path}")
    
    def plot_training_curves(self):
        """Generate and save training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Total loss
        ax = axes[0, 0]
        ax.plot(epochs, [l['total'] for l in self.train_losses], label='Train')
        ax.plot(epochs, [l['total'] for l in self.val_losses], label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True)
        
        # Coverage loss
        ax = axes[0, 1]
        ax.plot(epochs, [l['coverage'] for l in self.train_losses], label='Train')
        ax.plot(epochs, [l['coverage'] for l in self.val_losses], label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Coverage Loss')
        ax.set_title('Coverage Loss (Safety)')
        ax.legend()
        ax.grid(True)
        
        # Efficiency loss
        ax = axes[1, 0]
        ax.plot(epochs, [l['efficiency'] for l in self.train_losses], label='Train')
        ax.plot(epochs, [l['efficiency'] for l in self.val_losses], label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Efficiency Loss')
        ax.set_title('Efficiency Loss')
        ax.legend()
        ax.grid(True)
        
        # Smoothness loss
        ax = axes[1, 1]
        ax.plot(epochs, [l['smoothness'] for l in self.train_losses], label='Train')
        ax.plot(epochs, [l['smoothness'] for l in self.val_losses], label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Smoothness Loss')
        ax.set_title('Smoothness Loss')
        ax.legend()
        ax.grid(True)
        
        plt.suptitle('Learnable CP Training Curves')
        plt.tight_layout()
        
        # Save figure
        path = self.results_dir / 'training_curves.png'
        plt.savefig(path, dpi=150)
        plt.close()
        
        logging.info(f"Saved training curves to {path}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Train model
    trainer = LearnableCPTrainer()
    trainer.train()
    
    logging.info("Training complete!")