import json
import os
import sys
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

from utils import PATH_MODELS


def plot_training_curves(model_name, output_path=None):
    """
    Plot training curves (LAS) from scores.history.json
    
    Args:
        model_name: Name of the model (e.g., "UD_French-ALTS@2.18")
        output_path: Optional path to save the plot image
    """
    model_folder = os.path.join(PATH_MODELS, model_name)
    scores_file = os.path.join(model_folder, "scores.history.json")
    
    if not os.path.exists(scores_file):
        print(f"Error: {scores_file} not found!")
        sys.exit(1)
    
    with open(scores_file, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    if not history:
        print("Error: Empty history file")
        sys.exit(1)
    
    epochs = list(range(len(history)))
    
    las_epoch = []
    las_chuliu_epoch = []
    
    for epoch_data in history:
        las_epoch.append(epoch_data.get('LAS_epoch', None))
        las_chuliu_epoch.append(epoch_data.get('LAS_chuliu_epoch', None))
    
    las_valid = [x for x in las_epoch if x is not None]
    las_chuliu_valid = [x for x in las_chuliu_epoch if x is not None]
    
    if las_valid:
        las_max = max(las_valid)
        las_max_epoch = las_valid.index(las_max)
    if las_chuliu_valid:
        las_chuliu_max = max(las_chuliu_valid)
        las_chuliu_max_epoch = las_chuliu_valid.index(las_chuliu_max)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(epochs, las_epoch, marker='o', label='LAS', linewidth=2.5, markersize=7, color='#1f77b4')
    
    y_min = 0.85 if las_valid and min(las_valid) > 0.80 else 0.80
    ax.set_ylim(y_min, 1.0)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('LAS Score', fontsize=12)
    ax.set_title(f'Training Progress - Detailed View (≥{y_min:.2f}) - {model_name}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='lower right')
    
    if las_valid:
        ax.axhline(y=las_max, color='#1f77b4', linestyle=':', alpha=0.5, label=f'LAS max: {las_max:.4f} @ epoch {las_max_epoch}')
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = os.path.join(model_folder, "training_curves.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {output_path}")
    
    print("\n" + "="*60)
    print(f"TRAINING STATISTICS FOR: {model_name}")
    print("="*60)
    
    if las_valid:
        print(f"\nLAS (Standard Algorithm):")
        print(f"  • Best score: {max(las_valid):.4f} (epoch {las_epoch.index(max(las_valid))})")
        print(f"  • Final score: {las_epoch[-1]:.4f}")
        print(f"  • Average: {sum(las_valid)/len(las_valid):.4f}")
        print(f"  • Min: {min(las_valid):.4f}")
    
    print("="*60 + "\n")
    
    plt.show()


def list_available_models():
    """List all available trained models"""
    print("Available models:")
    for model_name in os.listdir(PATH_MODELS):
        model_path = os.path.join(PATH_MODELS, model_name)
        if os.path.isdir(model_path):
            scores_file = os.path.join(model_path, "scores.history.json")
            if os.path.exists(scores_file):
                print(f"  - {model_name} ✓")
            else:
                print(f"  - {model_name} (no scores)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot training curves from BertForDeprel training history'
    )
    parser.add_argument(
        'model_name',
        nargs='?',
        help='Name of the model (e.g., UD_French-ALTS@2.18)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output path for the plot image'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available models'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
    elif args.model_name:
        plot_training_curves(args.model_name, args.output)
    else:
        parser.print_help()
        print("\n" + "="*50)
        list_available_models()
