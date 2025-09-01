#!/usr/bin/env python3
"""
Dataset Visualization Script
Analyzes and visualizes the generated dataset to verify quality.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import seaborn as sns
from scipy import stats

class DatasetVisualizer:
    """Visualize and analyze the generated dataset."""
    
    def __init__(self, data_dir='data'):
        """Load dataset from files."""
        self.data_dir = data_dir
        
        # Load datasets
        train_data = np.load(f'{data_dir}/train_data.npz')
        cal_data = np.load(f'{data_dir}/calibration_data.npz')
        test_data = np.load(f'{data_dir}/test_data.npz')
        
        self.train_features = train_data['features']
        self.train_labels = train_data['labels']
        
        self.cal_features = cal_data['features']
        self.cal_labels = cal_data['labels']
        
        self.test_features = test_data['features']
        self.test_labels = test_data['labels']
        
        # Load metadata
        with open(f'{data_dir}/metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Load statistics
        with open(f'{data_dir}/dataset_stats.json', 'r') as f:
            self.stats = json.load(f)
        
        # Feature names
        self.feature_names = [
            'Dist to obstacle', 'Density 5m', 'Density 10m', 'Passage width',
            'Dist from start', 'Dist to goal', 'Curvature', 'Escape dirs',
            'Clearance var', 'Is narrow'
        ]
    
    def create_comprehensive_visualization(self):
        """Create comprehensive visualization of dataset quality."""
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Dataset composition
        ax1 = plt.subplot(3, 4, 1)
        self.plot_dataset_composition(ax1)
        
        # 2. Collision rate by noise level
        ax2 = plt.subplot(3, 4, 2)
        self.plot_collision_by_noise(ax2)
        
        # 3. Feature distributions
        ax3 = plt.subplot(3, 4, 3)
        self.plot_feature_importance(ax3)
        
        # 4. Distribution overlap check
        ax4 = plt.subplot(3, 4, 4)
        self.plot_distribution_overlap(ax4)
        
        # 5-8. Key feature distributions
        for i, feat_idx in enumerate([0, 3, 6, 9]):  # Key features
            ax = plt.subplot(3, 4, 5 + i)
            self.plot_feature_distribution(ax, feat_idx)
        
        # 9. Correlation matrix
        ax9 = plt.subplot(3, 4, 9)
        self.plot_correlation_matrix(ax9)
        
        # 10. Class separation
        ax10 = plt.subplot(3, 4, 10)
        self.plot_class_separation(ax10)
        
        # 11. Noise level balance
        ax11 = plt.subplot(3, 4, 11)
        self.plot_noise_balance(ax11)
        
        # 12. Sample quality metrics
        ax12 = plt.subplot(3, 4, 12)
        self.plot_quality_metrics(ax12)
        
        plt.suptitle('Dataset Quality Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        plt.savefig('results/dataset_analysis.png', dpi=150, bbox_inches='tight')
        print("Visualization saved to results/dataset_analysis.png")
        plt.show()
    
    def plot_dataset_composition(self, ax):
        """Plot dataset composition."""
        sizes = [len(self.train_features), len(self.cal_features), len(self.test_features)]
        labels = ['Train\n(60%)', 'Calibration\n(20%)', 'Test\n(20%)']
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                          autopct=lambda pct: f'{pct:.1f}%\n({int(pct*sum(sizes)/100)})',
                                          startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Dataset Composition')
    
    def plot_collision_by_noise(self, ax):
        """Plot collision rate by noise level."""
        noise_levels = []
        collisions = []
        
        for split_name, metadata in [('Train', self.metadata['train']),
                                    ('Cal', self.metadata['calibration']),
                                    ('Test', self.metadata['test'])]:
            noise_dict = {}
            collision_dict = {}
            
            for meta in metadata:
                level = meta['noise_level']
                if level not in noise_dict:
                    noise_dict[level] = 0
                    collision_dict[level] = 0
                
                noise_dict[level] += 1
                if meta['collision']:
                    collision_dict[level] += 1
            
            for level in sorted(noise_dict.keys()):
                noise_levels.append(f'σ={level}\n{split_name}')
                rate = 100 * collision_dict[level] / noise_dict[level] if noise_dict[level] > 0 else 0
                collisions.append(rate)
        
        colors = ['#2E86AB'] * 3 + ['#A23B72'] * 3 + ['#F18F01'] * 3
        bars = ax.bar(range(len(noise_levels)), collisions, color=colors)
        ax.set_xticks(range(len(noise_levels)))
        ax.set_xticklabels(noise_levels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Collision Rate (%)')
        ax.set_title('Collision Rate by Noise Level')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, collisions):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}%', ha='center', fontsize=8)
    
    def plot_feature_importance(self, ax):
        """Plot feature importance based on collision correlation."""
        importances = []
        
        for i in range(10):
            # Calculate point-biserial correlation
            all_features = np.vstack([self.train_features, self.cal_features, self.test_features])
            all_labels = np.hstack([self.train_labels, self.cal_labels, self.test_labels])
            
            corr, _ = stats.pointbiserialr(all_labels, all_features[:, i])
            importances.append(abs(corr))
        
        indices = np.argsort(importances)[::-1]
        
        bars = ax.barh(range(10), [importances[i] for i in indices],
                       color='#2E86AB')
        ax.set_yticks(range(10))
        ax.set_yticklabels([self.feature_names[i] for i in indices], fontsize=8)
        ax.set_xlabel('Absolute Correlation with Collision')
        ax.set_title('Feature Importance')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, [importances[j] for j in indices])):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=8)
    
    def plot_distribution_overlap(self, ax):
        """Check distribution overlap between splits."""
        # Calculate KL divergence between splits
        kl_divs = []
        
        for i in range(10):
            # Discretize features
            bins = np.linspace(0, 1, 20)
            
            train_hist, _ = np.histogram(self.train_features[:, i], bins=bins, density=True)
            cal_hist, _ = np.histogram(self.cal_features[:, i], bins=bins, density=True)
            test_hist, _ = np.histogram(self.test_features[:, i], bins=bins, density=True)
            
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            train_hist = train_hist + eps
            cal_hist = cal_hist + eps
            test_hist = test_hist + eps
            
            # Normalize
            train_hist = train_hist / train_hist.sum()
            cal_hist = cal_hist / cal_hist.sum()
            test_hist = test_hist / test_hist.sum()
            
            # Calculate KL divergences
            kl_train_cal = stats.entropy(train_hist, cal_hist)
            kl_train_test = stats.entropy(train_hist, test_hist)
            kl_cal_test = stats.entropy(cal_hist, test_hist)
            
            kl_divs.append(max(kl_train_cal, kl_train_test, kl_cal_test))
        
        bars = ax.bar(range(10), kl_divs, color='#A23B72')
        ax.set_xticks(range(10))
        ax.set_xticklabels([f'F{i}' for i in range(10)])
        ax.set_ylabel('Max KL Divergence')
        ax.set_title('Distribution Shift Check\n(Lower is better)')
        ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Threshold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    def plot_feature_distribution(self, ax, feature_idx):
        """Plot distribution of a specific feature."""
        train_feat = self.train_features[:, feature_idx]
        cal_feat = self.cal_features[:, feature_idx]
        test_feat = self.test_features[:, feature_idx]
        
        # Separate by collision
        train_coll = train_feat[self.train_labels == 1]
        train_safe = train_feat[self.train_labels == 0]
        
        bins = np.linspace(0, 1, 30)
        
        ax.hist(train_safe, bins=bins, alpha=0.5, label='Safe', color='green', density=True)
        ax.hist(train_coll, bins=bins, alpha=0.5, label='Collision', color='red', density=True)
        
        ax.set_xlabel(self.feature_names[feature_idx], fontsize=8)
        ax.set_ylabel('Density')
        ax.set_title(f'{self.feature_names[feature_idx]}', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    def plot_correlation_matrix(self, ax):
        """Plot feature correlation matrix."""
        all_features = np.vstack([self.train_features, self.cal_features])
        corr_matrix = np.corrcoef(all_features.T)
        
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels([f'F{i}' for i in range(10)], fontsize=8)
        ax.set_yticklabels([f'F{i}' for i in range(10)], fontsize=8)
        ax.set_title('Feature Correlations')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def plot_class_separation(self, ax):
        """Plot class separation using first two features."""
        # Combine train and cal
        all_features = np.vstack([self.train_features, self.cal_features])
        all_labels = np.hstack([self.train_labels, self.cal_labels])
        
        # Use first two features for visualization
        features_2d = all_features[:, :2]
        
        # Plot
        scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                           c=all_labels, cmap='RdYlGn_r', alpha=0.5, s=1)
        ax.set_xlabel('Feature 0: Nearest Obstacle Distance')
        ax.set_ylabel('Feature 1: Local Obstacle Density')
        ax.set_title('Class Separation (First 2 Features)')
        plt.colorbar(scatter, ax=ax, label='Collision')
    
    def plot_noise_balance(self, ax):
        """Plot noise level balance across splits."""
        splits = ['Train', 'Cal', 'Test']
        noise_levels = [0.1, 0.3, 0.5]
        
        data = []
        for metadata in [self.metadata['train'], self.metadata['calibration'], 
                        self.metadata['test']]:
            noise_counts = {level: 0 for level in noise_levels}
            for meta in metadata:
                noise_counts[meta['noise_level']] += 1
            
            total = sum(noise_counts.values())
            data.append([100 * noise_counts[level] / total for level in noise_levels])
        
        x = np.arange(len(splits))
        width = 0.25
        
        for i, level in enumerate(noise_levels):
            values = [d[i] for d in data]
            ax.bar(x + i * width, values, width, label=f'σ={level}')
        
        ax.set_xlabel('Split')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Noise Level Balance')
        ax.set_xticks(x + width)
        ax.set_xticklabels(splits)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    def plot_quality_metrics(self, ax):
        """Plot dataset quality metrics."""
        total_samples = len(self.train_features) + len(self.cal_features) + len(self.test_features)
        collision_rate = 100 * np.mean(np.hstack([self.train_labels, self.cal_labels, self.test_labels]))
        feature_dim = self.train_features.shape[1]
        min_samples = min(len(self.train_features), len(self.cal_features), len(self.test_features))
        
        # Create text display
        text = "Dataset Quality Metrics\n" + "="*25 + "\n"
        text += f"Total Samples: {total_samples:,}\n"
        text += f"Collision Rate: {collision_rate:.1f}%\n"
        text += f"Feature Dimension: {feature_dim}\n"
        text += f"Min Split Size: {min_samples:,}\n"
        text += "\n"
        
        # Add quality assessment
        quality_score = 0
        if total_samples > 5000:
            quality_score += 25
            text += "✓ Good sample size\n"
        else:
            text += "⚠ More samples recommended\n"
        
        if 15 < collision_rate < 40:
            quality_score += 25
            text += "✓ Balanced collision rate\n"
        else:
            text += "⚠ Imbalanced classes\n"
        
        if min_samples > 500:
            quality_score += 25
            text += "✓ Adequate split sizes\n"
        else:
            text += "⚠ Small split detected\n"
        
        # Check distribution shift using simple mean/std comparison
        max_shift = 0
        for i in range(10):
            train_mean = np.mean(self.train_features[:, i])
            test_mean = np.mean(self.test_features[:, i])
            train_std = np.std(self.train_features[:, i]) + 1e-8
            shift = abs(train_mean - test_mean) / train_std
            max_shift = max(max_shift, shift)
        
        if max_shift < 0.5:
            quality_score += 25
            text += "✓ No distribution shift\n"
        else:
            text += "⚠ Distribution shift detected\n"
        
        text += f"\nQuality Score: {quality_score}/100"
        
        ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=10,
               verticalalignment='center', family='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def generate_report(self):
        """Generate text report on dataset quality."""
        print("\n" + "="*60)
        print("DATASET QUALITY REPORT")
        print("="*60)
        
        # Basic statistics
        print("\n1. BASIC STATISTICS")
        print("-"*40)
        print(f"Total samples: {len(self.train_features) + len(self.cal_features) + len(self.test_features)}")
        print(f"Feature dimension: {self.train_features.shape[1]}")
        print(f"Train set: {len(self.train_features)} samples")
        print(f"Calibration set: {len(self.cal_features)} samples")
        print(f"Test set: {len(self.test_features)} samples")
        
        # Collision rates
        print("\n2. COLLISION RATES")
        print("-"*40)
        print(f"Overall: {100*np.mean(np.hstack([self.train_labels, self.cal_labels, self.test_labels])):.1f}%")
        print(f"Train: {100*np.mean(self.train_labels):.1f}%")
        print(f"Calibration: {100*np.mean(self.cal_labels):.1f}%")
        print(f"Test: {100*np.mean(self.test_labels):.1f}%")
        
        # Feature statistics
        print("\n3. FEATURE STATISTICS")
        print("-"*40)
        print("Feature ranges (min-max across all splits):")
        all_features = np.vstack([self.train_features, self.cal_features, self.test_features])
        for i, name in enumerate(self.feature_names):
            min_val = np.min(all_features[:, i])
            max_val = np.max(all_features[:, i])
            mean_val = np.mean(all_features[:, i])
            print(f"  {name:15s}: [{min_val:.3f}, {max_val:.3f}], mean={mean_val:.3f}")
        
        # Quality assessment
        print("\n4. QUALITY ASSESSMENT")
        print("-"*40)
        
        issues = []
        warnings = []
        
        # Check sample size
        if len(all_features) < 5000:
            warnings.append("Dataset size is small. Consider generating more samples.")
        
        # Check class balance
        collision_rate = np.mean(np.hstack([self.train_labels, self.cal_labels, self.test_labels]))
        if collision_rate < 0.1:
            issues.append("Very low collision rate. Model may not learn collision patterns.")
        elif collision_rate > 0.5:
            issues.append("Very high collision rate. Consider adjusting noise levels.")
        elif collision_rate < 0.15 or collision_rate > 0.4:
            warnings.append("Collision rate is slightly imbalanced.")
        
        # Check split sizes
        if len(self.test_features) < 500:
            warnings.append("Test set is small. Results may have high variance.")
        
        # Check feature coverage
        for i in range(10):
            if np.std(all_features[:, i]) < 0.05:
                warnings.append(f"Feature '{self.feature_names[i]}' has very low variance.")
        
        if issues:
            print("ISSUES (must address):")
            for issue in issues:
                print(f"  ❌ {issue}")
        
        if warnings:
            print("\nWARNINGS (consider addressing):")
            for warning in warnings:
                print(f"  ⚠️  {warning}")
        
        if not issues and not warnings:
            print("✅ Dataset quality is good!")
        
        # Recommendations
        print("\n5. RECOMMENDATIONS")
        print("-"*40)
        if collision_rate < 0.2:
            print("- Consider increasing noise levels to generate more challenging scenarios")
        if len(all_features) < 10000:
            print("- Generate more samples for better generalization")
        if not issues:
            print("- Dataset is ready for training learnable CP model")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    # Create visualizer
    visualizer = DatasetVisualizer('data')
    
    # Generate comprehensive visualization
    visualizer.create_comprehensive_visualization()
    
    # Generate text report
    visualizer.generate_report()