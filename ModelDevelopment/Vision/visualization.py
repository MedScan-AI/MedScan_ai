"""
Visualization and reporting utilities.
Generates HTML reports with training curves, confusion matrices, and model comparisons.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import base64
import io
import csv

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 encoded string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def plot_training_curves(epoch_metrics: List[Dict]) -> str:
    """Plot training curves for accuracy, precision, recall, F1-score, and AUC."""
    if not epoch_metrics:
        return ""
    
    epochs = [m['epoch'] for m in epoch_metrics]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training Metrics Across Epochs', fontsize=16, fontweight='bold')
    
    # Accuracy
    ax = axes[0, 0]
    ax.plot(epochs, [m.get('train_accuracy', 0) for m in epoch_metrics], 
            label='Train Accuracy', marker='o', linewidth=2)
    ax.plot(epochs, [m.get('val_accuracy', 0) for m in epoch_metrics], 
            label='Val Accuracy', marker='s', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Precision
    ax = axes[0, 1]
    ax.plot(epochs, [m.get('precision', 0) for m in epoch_metrics], 
            label='Precision', marker='o', linewidth=2, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision')
    ax.set_title('Precision')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Recall
    ax = axes[0, 2]
    ax.plot(epochs, [m.get('recall', 0) for m in epoch_metrics], 
            label='Recall', marker='o', linewidth=2, color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall')
    ax.set_title('Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # F1-Score
    ax = axes[1, 0]
    ax.plot(epochs, [m.get('f1_score', 0) for m in epoch_metrics], 
            label='F1-Score', marker='o', linewidth=2, color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # AUC
    ax = axes[1, 1]
    ax.plot(epochs, [m.get('auc', 0) for m in epoch_metrics], 
            label='AUC', marker='o', linewidth=2, color='purple')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.set_title('AUC (Area Under ROC Curve)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss
    ax = axes[1, 2]
    ax.plot(epochs, [m.get('train_loss', 0) for m in epoch_metrics], 
            label='Train Loss', marker='o', linewidth=2)
    ax.plot(epochs, [m.get('val_loss', 0) for m in epoch_metrics], 
            label='Val Loss', marker='s', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plot_to_base64(fig)


def plot_per_class_metrics(epoch_metrics: List[Dict], class_names: List[str]) -> str:
    """Plot per-class precision, recall, F1-score, and support across epochs."""
    if not epoch_metrics or not class_names:
        return ""
    
    epochs = [m['epoch'] for m in epoch_metrics]
    n_classes = len(class_names)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Per-Class Metrics Across Epochs', fontsize=16, fontweight='bold')
    
    # Per-class Precision
    ax = axes[0, 0]
    for i, class_name in enumerate(class_names):
        precisions = [m.get('per_class_precision', [0]*n_classes)[i] if i < len(m.get('per_class_precision', [])) else 0 
                     for m in epoch_metrics]
        ax.plot(epochs, precisions, label=class_name, marker='o', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision')
    ax.set_title('Per-Class Precision')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Per-class Recall
    ax = axes[0, 1]
    for i, class_name in enumerate(class_names):
        recalls = [m.get('per_class_recall', [0]*n_classes)[i] if i < len(m.get('per_class_recall', [])) else 0 
                  for m in epoch_metrics]
        ax.plot(epochs, recalls, label=class_name, marker='s', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall')
    ax.set_title('Per-Class Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Per-class F1-Score
    ax = axes[1, 0]
    for i, class_name in enumerate(class_names):
        f1_scores = [m.get('per_class_f1', [0]*n_classes)[i] if i < len(m.get('per_class_f1', [])) else 0 
                    for m in epoch_metrics]
        ax.plot(epochs, f1_scores, label=class_name, marker='^', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1-Score')
    ax.set_title('Per-Class F1-Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Per-class Support (bar chart for last epoch)
    ax = axes[1, 1]
    if epoch_metrics:
        last_epoch = epoch_metrics[-1]
        supports = last_epoch.get('per_class_support', [0]*n_classes)
        if len(supports) == n_classes:
            bars = ax.bar(class_names, supports, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'][:n_classes])
            ax.set_xlabel('Class')
            ax.set_ylabel('Support (Number of Samples)')
            ax.set_title('Per-Class Support (Last Epoch)')
            ax.grid(True, alpha=0.3, axis='y')
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plot_to_base64(fig)


def plot_hyperparameter_tracking(epoch_metrics: List[Dict]) -> str:
    """Plot hyperparameter values (e.g., learning rate) across epochs."""
    if not epoch_metrics:
        return ""
    
    epochs = [m['epoch'] for m in epoch_metrics]
    
    # Find hyperparameters that vary across epochs
    hyperparams = {}
    for metric in epoch_metrics:
        for key, value in metric.items():
            if key not in ['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy',
                          'precision', 'recall', 'f1_score', 'accuracy', 'auc',
                          'per_class_precision', 'per_class_recall', 'per_class_f1', 
                          'per_class_support', 'confusion_matrix', 'classification_report']:
                if isinstance(value, (int, float)):
                    if key not in hyperparams:
                        hyperparams[key] = []
                    hyperparams[key].append(value)
    
    if not hyperparams:
        return ""
    
    n_params = len(hyperparams)
    n_cols = min(2, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6*n_rows))
    fig.suptitle('Hyperparameter Tracking Across Epochs', fontsize=16, fontweight='bold')
    
    # Convert axes to a flat list for consistent handling
    # plt.subplots returns different types depending on the grid shape
    if n_params == 1:
        # Single subplot: axes is a single Axes object
        axes = [axes]
    else:
        # Multiple subplots: axes is a numpy array (1D or 2D)
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
            # Convert numpy array to list to ensure proper Axes object access
            axes = [ax for ax in axes]
        else:
            # If it's already a list or single object, convert to list
            axes = [axes] if not isinstance(axes, list) else axes
    
    for idx, (param_name, values) in enumerate(hyperparams.items()):
        if idx < len(axes):
            ax = axes[idx]
            ax.plot(epochs[:len(values)], values, marker='o', linewidth=2, color='#e74c3c')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(param_name.replace('_', ' ').title())
            ax.set_title(f'{param_name.replace("_", " ").title()} Over Time')
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_params, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return plot_to_base64(fig)


def plot_confusion_matrix(cm: List[List[int]], class_names: List[str]) -> str:
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cm_array = np.array(cm)
    sns.heatmap(
        cm_array, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Count'}
    )
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return plot_to_base64(fig)


def plot_metrics_comparison(models_data: List[Dict]) -> str:
    """Plot bar chart comparing metrics across different models."""
    if not models_data:
        return ""
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    model_names = [m['model_name'] for m in models_data]
    
    # Prepare data
    data = {metric: [m.get('metrics', {}).get(metric, 0) for m in models_data] 
            for metric in metrics}
    
    x = np.arange(len(model_names))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, metric in enumerate(metrics):
        offset = (i - len(metrics) / 2) * width + width / 2
        ax.bar(x + offset, data[metric], width, label=metric.replace('_', ' ').title())
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    return plot_to_base64(fig)


def plot_hyperparameter_impact(hyperparameter_data: List[Dict]) -> str:
    """Plot hyperparameter impact on model performance."""
    if not hyperparameter_data or len(hyperparameter_data) < 2:
        return ""
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(hyperparameter_data)
    
    # Get numeric hyperparameters
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove metrics columns
    metric_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'val_accuracy', 'val_loss']
    hyperparam_cols = [col for col in numeric_cols if col not in metric_cols and col != 'epoch']
    
    if not hyperparam_cols:
        return ""
    
    # Plot impact of each hyperparameter
    n_params = min(len(hyperparam_cols), 4)  # Limit to 4 most important
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Hyperparameter Impact on Model Performance', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, param in enumerate(hyperparam_cols[:4]):
        ax = axes[idx]
        
        # Scatter plot: hyperparameter vs accuracy
        ax.scatter(df[param], df.get('val_accuracy', df.get('accuracy', 0)), 
                  alpha=0.6, s=100)
        ax.set_xlabel(param.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_title(f'{param.replace("_", " ").title()} vs Accuracy', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(hyperparam_cols[:4]), 4):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return plot_to_base64(fig)


def generate_training_report(
    model_name: str,
    dataset_name: str,
    epoch_metrics: List[Dict],
    test_metrics: Dict,
    class_names: List[str],
    hyperparameters: Optional[Dict] = None,
    output_path: Path = None
) -> str:
    """
    Generate comprehensive HTML training report.
    
    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        epoch_metrics: List of metrics per epoch
        test_metrics: Final test set metrics
        class_names: List of class names
        hyperparameters: Optional hyperparameter values
        output_path: Path to save HTML report
        
    Returns:
        Path to generated HTML report
    """
    logger.info(f"Generating training report for {model_name}...")
    
    # Generate visualizations
    training_curves_img = plot_training_curves(epoch_metrics) if epoch_metrics else ""
    
    per_class_metrics_img = ""
    if epoch_metrics and class_names:
        try:
            per_class_metrics_img = plot_per_class_metrics(epoch_metrics, class_names)
        except Exception as e:
            logger.warning(f"Could not generate per-class metrics plot: {e}")
            per_class_metrics_img = ""
    
    hyperparameter_tracking_img = ""
    if epoch_metrics:
        try:
            hyperparameter_tracking_img = plot_hyperparameter_tracking(epoch_metrics)
        except Exception as e:
            logger.warning(f"Could not generate hyperparameter tracking plot: {e}")
            hyperparameter_tracking_img = ""
    
    confusion_matrix_img = ""
    if test_metrics.get('confusion_matrix'):
        try:
            confusion_matrix_img = plot_confusion_matrix(
                test_metrics['confusion_matrix'],
                class_names
            )
        except Exception as e:
            logger.warning(f"Could not generate confusion matrix plot: {e}")
            confusion_matrix_img = ""
    
    # Get final metrics
    final_metrics = test_metrics.copy()
    if epoch_metrics:
        final_epoch = epoch_metrics[-1]
        final_metrics.update({
            'final_train_accuracy': final_epoch.get('train_accuracy', 0),
            'final_val_accuracy': final_epoch.get('val_accuracy', 0),
            'final_train_loss': final_epoch.get('train_loss', 0),
            'final_val_loss': final_epoch.get('val_loss', 0)
        })
    
    # Create HTML content
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Training Report - {model_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .metric-card .value {{
            font-size: 32px;
            font-weight: bold;
            margin: 0;
        }}
        .plot-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .info-section {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Training Report: {model_name}</h1>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        
        <div class="info-section">
            <strong>Dataset:</strong> {dataset_name}<br>
            <strong>Model:</strong> {model_name}<br>
            <strong>Number of Classes:</strong> {len(class_names)}<br>
            <strong>Classes:</strong> {', '.join(class_names)}
        </div>
        
        <h2>Final Test Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Accuracy</h3>
                <p class="value">{final_metrics.get('accuracy', 0):.4f}</p>
            </div>
            <div class="metric-card">
                <h3>Precision</h3>
                <p class="value">{final_metrics.get('precision', 0):.4f}</p>
            </div>
            <div class="metric-card">
                <h3>Recall</h3>
                <p class="value">{final_metrics.get('recall', 0):.4f}</p>
            </div>
            <div class="metric-card">
                <h3>F1-Score</h3>
                <p class="value">{final_metrics.get('f1_score', 0):.4f}</p>
            </div>
            <div class="metric-card">
                <h3>AUC</h3>
                <p class="value">{final_metrics.get('auc', 0):.4f}</p>
            </div>
        </div>
"""
    
    # Add training curves
    if training_curves_img:
        html_content += f"""
        <h2>Training Curves</h2>
        <div class="plot-container">
            <img src="data:image/png;base64,{training_curves_img}" alt="Training Curves">
        </div>
"""
    
    # Add per-class metrics
    if per_class_metrics_img:
        html_content += f"""
        <h2>Per-Class Metrics Across Epochs</h2>
        <p>This visualization shows precision, recall, F1-score, and support for each class across all training epochs.</p>
        <div class="plot-container">
            <img src="data:image/png;base64,{per_class_metrics_img}" alt="Per-Class Metrics">
        </div>
"""
    
    # Add hyperparameter tracking
    if hyperparameter_tracking_img:
        html_content += f"""
        <h2>Hyperparameter Tracking</h2>
        <p>This visualization shows how hyperparameters (e.g., learning rate) change across training epochs.</p>
        <div class="plot-container">
            <img src="data:image/png;base64,{hyperparameter_tracking_img}" alt="Hyperparameter Tracking">
        </div>
"""
    
    # Add confusion matrix
    if confusion_matrix_img:
        html_content += f"""
        <h2>Confusion Matrix</h2>
        <div class="plot-container">
            <img src="data:image/png;base64,{confusion_matrix_img}" alt="Confusion Matrix">
        </div>
"""
    
    # Add per-class metrics table with support
    if test_metrics.get('per_class_precision'):
        html_content += """
        <h2>Per-Class Metrics (Test Set)</h2>
        <table>
            <thead>
                <tr>
                    <th>Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Support</th>
                </tr>
            </thead>
            <tbody>
"""
        for i, class_name in enumerate(class_names):
            precision = test_metrics.get('per_class_precision', [0])[i] if i < len(test_metrics.get('per_class_precision', [])) else 0
            recall = test_metrics.get('per_class_recall', [0])[i] if i < len(test_metrics.get('per_class_recall', [])) else 0
            f1 = test_metrics.get('per_class_f1', [0])[i] if i < len(test_metrics.get('per_class_f1', [])) else 0
            support = test_metrics.get('per_class_support', [0])[i] if i < len(test_metrics.get('per_class_support', [])) else 0
            html_content += f"""
                <tr>
                    <td>{class_name}</td>
                    <td>{precision:.4f}</td>
                    <td>{recall:.4f}</td>
                    <td>{f1:.4f}</td>
                    <td>{int(support)}</td>
                </tr>
"""
        html_content += """
            </tbody>
        </table>
"""
    
    # Add epoch-wise metrics table
    if epoch_metrics:
        html_content += """
        <h2>Epoch-Wise Metrics</h2>
        <div style="overflow-x: auto;">
        <table>
            <thead>
                <tr>
                    <th>Epoch</th>
                    <th>Train Acc</th>
                    <th>Val Acc</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>AUC</th>
"""
        # Add hyperparameter columns if available
        if epoch_metrics and 'learning_rate' in epoch_metrics[0]:
            html_content += "<th>Learning Rate</th>"
        html_content += """
                </tr>
            </thead>
            <tbody>
"""
        for epoch_metric in epoch_metrics:
            html_content += f"""
                <tr>
                    <td>{epoch_metric.get('epoch', 0)}</td>
                    <td>{epoch_metric.get('train_accuracy', 0):.4f}</td>
                    <td>{epoch_metric.get('val_accuracy', 0):.4f}</td>
                    <td>{epoch_metric.get('precision', 0):.4f}</td>
                    <td>{epoch_metric.get('recall', 0):.4f}</td>
                    <td>{epoch_metric.get('f1_score', 0):.4f}</td>
                    <td>{epoch_metric.get('auc', 0):.4f}</td>
"""
            if 'learning_rate' in epoch_metric:
                html_content += f"<td>{epoch_metric.get('learning_rate', 0):.6f}</td>"
            html_content += """
                </tr>
"""
        html_content += """
            </tbody>
        </table>
        </div>
"""
    
    # Add hyperparameters
    if hyperparameters:
        html_content += """
        <h2>Hyperparameters</h2>
        <table>
            <thead>
                <tr>
                    <th>Hyperparameter</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
"""
        for key, value in hyperparameters.items():
            html_content += f"""
                <tr>
                    <td>{key.replace('_', ' ').title()}</td>
                    <td>{value}</td>
                </tr>
"""
        html_content += """
            </tbody>
        </table>
"""
        
        # Add hyperparameter impact analysis if we have epoch metrics
        if epoch_metrics and len(epoch_metrics) > 1:
            try:
                # Prepare data for hyperparameter impact analysis
                hyperparam_data = []
                for epoch_metric in epoch_metrics:
                    data_point = {
                        'epoch': epoch_metric.get('epoch', 0),
                        'val_accuracy': epoch_metric.get('val_accuracy', 0),
                        'accuracy': epoch_metric.get('accuracy', 0),
                        'precision': epoch_metric.get('precision', 0),
                        'recall': epoch_metric.get('recall', 0),
                        'f1_score': epoch_metric.get('f1_score', 0),
                        'auc': epoch_metric.get('auc', 0)
                    }
                    # Add hyperparameters if available
                    for key, value in hyperparameters.items():
                        data_point[key] = value
                    hyperparam_data.append(data_point)
                
                if hyperparam_data:
                    impact_img = plot_hyperparameter_impact(hyperparam_data)
                    if impact_img:
                        html_content += f"""
        <h2>Hyperparameter Impact Analysis</h2>
        <p>This analysis shows how hyperparameters affect model performance across training epochs.</p>
        <div class="plot-container">
            <img src="data:image/png;base64,{impact_img}" alt="Hyperparameter Impact">
        </div>
"""
            except Exception as e:
                logger.warning(f"Could not generate hyperparameter impact analysis: {e}")
    
    html_content += """
    </div>
</body>
</html>
"""
    
    # Save HTML file
    if output_path is None:
        output_path = Path(f"{model_name}_training_report.html")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Training report saved to {output_path}")
    return str(output_path)


def generate_model_comparison_report(
    models_data: List[Dict],
    output_path: Path
) -> str:
    """
    Generate HTML report comparing multiple models.
    
    Args:
        models_data: List of dictionaries with model information and metrics
        output_path: Path to save HTML report
        
    Returns:
        Path to generated HTML report
    """
    logger.info("Generating model comparison report...")
    
    # Generate comparison plot
    comparison_img = plot_metrics_comparison(models_data)
    
    # Create HTML content
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Model Comparison Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .plot-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .best {{
            background-color: #d4edda;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Model Comparison Report</h1>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
"""
    
    if comparison_img:
        html_content += f"""
        <div class="plot-container">
            <img src="data:image/png;base64,{comparison_img}" alt="Model Comparison">
        </div>
"""
    
    # Create comparison table
    html_content += """
        <h2>Detailed Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>AUC</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # Find best model for each metric
    best_accuracy = max(m.get('metrics', {}).get('accuracy', 0) for m in models_data)
    best_precision = max(m.get('metrics', {}).get('precision', 0) for m in models_data)
    best_recall = max(m.get('metrics', {}).get('recall', 0) for m in models_data)
    best_f1 = max(m.get('metrics', {}).get('f1_score', 0) for m in models_data)
    best_auc = max(m.get('metrics', {}).get('auc', 0) for m in models_data)
    
    for model_data in models_data:
        metrics = model_data.get('metrics', {})
        accuracy = metrics.get('accuracy', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        f1 = metrics.get('f1_score', 0)
        auc = metrics.get('auc', 0)
        
        # Determine which metrics are best
        row_class = ""
        if accuracy == best_accuracy or precision == best_precision or \
           recall == best_recall or f1 == best_f1 or auc == best_auc:
            row_class = "best"
        
        html_content += f"""
                <tr class="{row_class}">
                    <td>{model_data.get('model_name', 'Unknown')}</td>
                    <td>{accuracy:.4f}</td>
                    <td>{precision:.4f}</td>
                    <td>{recall:.4f}</td>
                    <td>{f1:.4f}</td>
                    <td>{auc:.4f}</td>
                </tr>
"""
    
    html_content += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    
    # Save HTML file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Model comparison report saved to {output_path}")
    return str(output_path)


def export_epoch_metrics_to_csv(
    epoch_metrics: List[Dict],
    class_names: List[str],
    output_path: Path
) -> str:
    """
    Export epoch-wise metrics to CSV file.
    
    Args:
        epoch_metrics: List of metrics per epoch
        class_names: List of class names
        output_path: Path to save CSV file
        
    Returns:
        Path to generated CSV file
    """
    if not epoch_metrics:
        logger.warning("No epoch metrics to export")
        return ""
    
    logger.info(f"Exporting epoch metrics to CSV: {output_path}")
    
    # Prepare data for CSV
    rows = []
    for epoch_metric in epoch_metrics:
        row = {
            'trial': epoch_metric.get('trial', 1),  # Include trial number
            'epoch': epoch_metric.get('epoch', 0),
            'train_loss': epoch_metric.get('train_loss', 0),
            'train_accuracy': epoch_metric.get('train_accuracy', 0),
            'val_loss': epoch_metric.get('val_loss', 0),
            'val_accuracy': epoch_metric.get('val_accuracy', 0),
            'precision': epoch_metric.get('precision', 0),
            'recall': epoch_metric.get('recall', 0),
            'f1_score': epoch_metric.get('f1_score', 0),
            'accuracy': epoch_metric.get('accuracy', 0),
            'auc': epoch_metric.get('auc', 0)
        }
        
        # Add hyperparameters (e.g., learning_rate)
        for key, value in epoch_metric.items():
            if key not in ['trial', 'epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy',
                          'precision', 'recall', 'f1_score', 'accuracy', 'auc',
                          'per_class_precision', 'per_class_recall', 'per_class_f1',
                          'per_class_support', 'confusion_matrix', 'classification_report']:
                if isinstance(value, (int, float)):
                    row[key] = value
        
        # Add per-class metrics
        per_class_precision = epoch_metric.get('per_class_precision', [])
        per_class_recall = epoch_metric.get('per_class_recall', [])
        per_class_f1 = epoch_metric.get('per_class_f1', [])
        per_class_support = epoch_metric.get('per_class_support', [])
        
        for i, class_name in enumerate(class_names):
            if i < len(per_class_precision):
                row[f'{class_name}_precision'] = per_class_precision[i]
            else:
                row[f'{class_name}_precision'] = 0.0
            
            if i < len(per_class_recall):
                row[f'{class_name}_recall'] = per_class_recall[i]
            else:
                row[f'{class_name}_recall'] = 0.0
            
            if i < len(per_class_f1):
                row[f'{class_name}_f1_score'] = per_class_f1[i]
            else:
                row[f'{class_name}_f1_score'] = 0.0
            
            if i < len(per_class_support):
                row[f'{class_name}_support'] = int(per_class_support[i])
            else:
                row[f'{class_name}_support'] = 0
        
        rows.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    logger.info(f"Epoch metrics CSV saved to {output_path} ({len(rows)} rows)")
    return str(output_path)

