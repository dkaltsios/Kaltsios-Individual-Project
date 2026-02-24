"""
Generate architecture diagrams for Stage 1 image models.
Creates presentation-ready diagrams showing pretrained backbone → modified layer → output.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def create_model_diagram(model_name, backbone_name, feature_dim, output_path, figsize=(12, 4)):
    """
    Create an architecture diagram for a CNN model.
    
    Args:
        model_name: Name of the model (e.g., "EfficientNet-B0")
        backbone_name: Name of the pretrained backbone
        feature_dim: Dimension of features before classifier (e.g., 1280 for EfficientNet-B0)
        output_path: Path to save the diagram
        figsize: Figure size (width, height)
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    # Colors
    backbone_color = '#E3F2FD'  # Light blue
    classifier_color = '#C8E6C9'  # Light green
    output_color = '#FFF9C4'  # Light yellow
    
    # Box dimensions
    box_width = 2.2
    box_height = 1.5
    y_center = 1.5
    
    # Position boxes
    x_backbone = 1.5
    x_classifier = 4.5
    x_output = 7.5
    
    # 1. Pretrained Backbone
    backbone_box = FancyBboxPatch(
        (x_backbone - box_width/2, y_center - box_height/2),
        box_width, box_height,
        boxstyle="round,pad=0.1",
        facecolor=backbone_color,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(backbone_box)
    ax.text(x_backbone, y_center + 0.4, f'Pretrained {backbone_name}',
            ha='center', va='center', fontsize=11, weight='bold')
    ax.text(x_backbone, y_center, 'ImageNet Weights\n(IMAGENET1K_V1)',
            ha='center', va='center', fontsize=9)
    ax.text(x_backbone, y_center - 0.4, 'Feature Extraction',
            ha='center', va='center', fontsize=9, style='italic')
    
    # 2. Modified Classifier Layer
    classifier_box = FancyBboxPatch(
        (x_classifier - box_width/2, y_center - box_height/2),
        box_width, box_height,
        boxstyle="round,pad=0.1",
        facecolor=classifier_color,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(classifier_box)
    ax.text(x_classifier, y_center + 0.4, 'Modified Classifier',
            ha='center', va='center', fontsize=11, weight='bold')
    ax.text(x_classifier, y_center, f'Linear({feature_dim} → 1)',
            ha='center', va='center', fontsize=10, family='monospace')
    ax.text(x_classifier, y_center - 0.4, 'Trained from scratch',
            ha='center', va='center', fontsize=9, style='italic')
    
    # 3. Output
    output_box = FancyBboxPatch(
        (x_output - box_width/2, y_center - box_height/2),
        box_width, box_height,
        boxstyle="round,pad=0.1",
        facecolor=output_color,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(output_box)
    ax.text(x_output, y_center + 0.4, 'Output',
            ha='center', va='center', fontsize=11, weight='bold')
    ax.text(x_output, y_center, 'Single Logit\n(Binary Classification)',
            ha='center', va='center', fontsize=9)
    ax.text(x_output, y_center - 0.4, 'Sigmoid → Probability',
            ha='center', va='center', fontsize=9, style='italic')
    
    # Arrows
    arrow1 = FancyArrowPatch(
        (x_backbone + box_width/2, y_center),
        (x_classifier - box_width/2, y_center),
        arrowstyle='->', mutation_scale=20, linewidth=2.5,
        color='black', zorder=3
    )
    ax.add_patch(arrow1)
    
    arrow2 = FancyArrowPatch(
        (x_classifier + box_width/2, y_center),
        (x_output - box_width/2, y_center),
        arrowstyle='->', mutation_scale=20, linewidth=2.5,
        color='black', zorder=3
    )
    ax.add_patch(arrow2)
    
    # Title
    ax.text(5, 2.6, f'{model_name} Architecture', 
            ha='center', va='center', fontsize=14, weight='bold')
    
    # Input annotation
    ax.text(0.5, y_center, 'Input:\n224×224×3\nRGB Image',
            ha='center', va='center', fontsize=9, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray'))
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved diagram to: {output_path}")
    plt.close()


def create_combined_diagram(output_path='Stage1/architecture_diagrams/combined_architectures.png'):
    """Create a side-by-side comparison of both architectures."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    
    for ax, model_name, backbone_name, feature_dim in [
        (ax1, 'EfficientNet-B0', 'EfficientNet-B0', 1280),
        (ax2, 'ResNet50', 'ResNet50', 2048)
    ]:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 3)
        ax.axis('off')
        
        # Colors
        backbone_color = '#E3F2FD'
        classifier_color = '#C8E6C9'
        output_color = '#FFF9C4'
        
        # Box dimensions
        box_width = 2.2
        box_height = 1.5
        y_center = 1.5
        
        # Position boxes
        x_backbone = 2.5
        x_classifier = 5.0
        x_output = 7.5
        
        # 1. Pretrained Backbone
        backbone_box = FancyBboxPatch(
            (x_backbone - box_width/2, y_center - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.1",
            facecolor=backbone_color,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(backbone_box)
        ax.text(x_backbone, y_center + 0.4, f'Pretrained\n{backbone_name}',
                ha='center', va='center', fontsize=10, weight='bold')
        ax.text(x_backbone, y_center, 'ImageNet\nWeights',
                ha='center', va='center', fontsize=8)
        ax.text(x_backbone, y_center - 0.4, f'{feature_dim}-dim\nfeatures',
                ha='center', va='center', fontsize=8, style='italic')
        
        # 2. Modified Classifier Layer
        classifier_box = FancyBboxPatch(
            (x_classifier - box_width/2, y_center - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.1",
            facecolor=classifier_color,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(classifier_box)
        ax.text(x_classifier, y_center + 0.4, 'Modified\nClassifier',
                ha='center', va='center', fontsize=10, weight='bold')
        ax.text(x_classifier, y_center, f'Linear({feature_dim}→1)',
                ha='center', va='center', fontsize=9, family='monospace')
        ax.text(x_classifier, y_center - 0.4, 'Trained',
                ha='center', va='center', fontsize=8, style='italic')
        
        # 3. Output
        output_box = FancyBboxPatch(
            (x_output - box_width/2, y_center - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.1",
            facecolor=output_color,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(output_box)
        ax.text(x_output, y_center + 0.4, 'Output',
                ha='center', va='center', fontsize=10, weight='bold')
        ax.text(x_output, y_center, 'Logit\n→ Sigmoid',
                ha='center', va='center', fontsize=8)
        ax.text(x_output, y_center - 0.4, 'Probability',
                ha='center', va='center', fontsize=8, style='italic')
        
        # Arrows
        arrow1 = FancyArrowPatch(
            (x_backbone + box_width/2, y_center),
            (x_classifier - box_width/2, y_center),
            arrowstyle='->', mutation_scale=20, linewidth=2,
            color='black', zorder=3
        )
        ax.add_patch(arrow1)
        
        arrow2 = FancyArrowPatch(
            (x_classifier + box_width/2, y_center),
            (x_output - box_width/2, y_center),
            arrowstyle='->', mutation_scale=20, linewidth=2,
            color='black', zorder=3
        )
        ax.add_patch(arrow2)
        
        # Title
        ax.text(5, 2.6, model_name, 
                ha='center', va='center', fontsize=12, weight='bold')
        
        # Input annotation
        ax.text(0.8, y_center, 'Input\n224×224×3',
                ha='center', va='center', fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))
    
    plt.suptitle('Stage 1: Image Model Architectures', fontsize=14, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved combined diagram to: {output_path}")
    plt.close()


if __name__ == "__main__":
    import os
    from pathlib import Path
    
    # Create output directory
    output_dir = Path("Stage1/architecture_diagrams")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate individual diagrams
    print("Generating architecture diagrams...")
    create_model_diagram(
        model_name="EfficientNet-B0",
        backbone_name="EfficientNet-B0",
        feature_dim=1280,
        output_path=str(output_dir / "efficientnet_b0_architecture.png")
    )
    
    create_model_diagram(
        model_name="ResNet50",
        backbone_name="ResNet50",
        feature_dim=2048,
        output_path=str(output_dir / "resnet50_architecture.png")
    )
    
    # Generate combined diagram
    create_combined_diagram(str(output_dir / "combined_architectures.png"))
    
    print("\n✓ All diagrams generated successfully!")
    print(f"  Output directory: {output_dir}")
