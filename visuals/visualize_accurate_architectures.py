"""
Generate accurate neural network architecture diagrams based on actual model structure.
Shows exact input dimensions, layer channels, and classifier dimensions.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np


def create_accurate_efficientnet_diagram(output_path):
    """
    Create accurate EfficientNet-B0 diagram showing:
    - Input: 224×224×3
    - Backbone stages with actual channel dimensions
    - Global Average Pooling → 1280 features
    - Classifier: Linear(1280 → 1)
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)

    y_center = 4.5
    node_radius = 0.2

    # Layer configurations with actual dimensions
    layers = [
        {
            "x": 1.5,
            "label": "Input Layer",
            "nodes": [
                {"label": "R\n224×224", "color": "#FF6B6B"},
                {"label": "G\n224×224", "color": "#4ECDC4"},
                {"label": "B\n224×224", "color": "#45B7D1"},
            ],
            "box_color": "#B3E5FC",
            "description": "RGB Image\n224×224×3"
        },
        {
            "x": 4,
            "label": "Initial Conv",
            "nodes": [
                {"label": "32", "color": "#81D4FA"},
            ] * 8,  # Show 8 representative nodes
            "box_color": "#E1BEE7",
            "description": "Conv2d(3→32)\nStride 2"
        },
        {
            "x": 6.5,
            "label": "MBConv Blocks",
            "nodes": [
                {"label": "16", "color": "#CE93D8"},
                {"label": "24", "color": "#CE93D8"},
                {"label": "40", "color": "#CE93D8"},
                {"label": "80", "color": "#CE93D8"},
                {"label": "112", "color": "#CE93D8"},
                {"label": "192", "color": "#CE93D8"},
            ],
            "box_color": "#D1C4E9",
            "description": "MBConv Blocks\n(7 stages)"
        },
        {
            "x": 9,
            "label": "Global Avg Pool",
            "nodes": [
                {"label": "1280", "color": "#A5D6A7"},
            ] * 5,  # Show 5 nodes for 1280-dim
            "box_color": "#C8E6C9",
            "description": "Global Avg Pool\n1280 features"
        },
        {
            "x": 11.5,
            "label": "Classifier",
            "nodes": [
                {"label": "1", "color": "#FFCC80"},
            ],
            "box_color": "#FFF9C4",
            "description": "Linear(1280→1)\nTrained"
        },
    ]

    all_node_positions = []

    for layer_idx, layer in enumerate(layers):
        x = layer["x"]
        n_nodes = len(layer["nodes"])
        layer_height = min(5, n_nodes * 0.6)
        y_positions = np.linspace(y_center + layer_height/2, y_center - layer_height/2, n_nodes)
        all_node_positions.append((x, y_positions, layer["nodes"]))

        # Layer label box
        label_box = FancyBboxPatch(
            (x - 0.7, y_center + layer_height/2 + 0.5),
            1.4, 0.8,
            boxstyle="round,pad=0.08",
            facecolor=layer["box_color"],
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(label_box)
        ax.text(x, y_center + layer_height/2 + 0.9, layer["label"],
                ha='center', va='center', fontsize=10, weight='bold')

        # Description below label
        ax.text(x, y_center + layer_height/2 + 0.3, layer["description"],
                ha='center', va='center', fontsize=8, style='italic')

        # Draw nodes
        for i, (y, node_info) in enumerate(zip(y_positions, layer["nodes"])):
            circle = Circle((x, y), node_radius,
                            facecolor=node_info["color"],
                            edgecolor='black',
                            linewidth=1.5)
            ax.add_patch(circle)
            ax.text(x, y, node_info["label"],
                    ha='center', va='center', fontsize=7, weight='bold')

    # Connections between layers
    for layer_idx in range(len(all_node_positions) - 1):
        x1, ys1, nodes1 = all_node_positions[layer_idx]
        x2, ys2, nodes2 = all_node_positions[layer_idx + 1]

        # Connect every node in layer i to every node in layer i+1
        for y1 in ys1:
            for y2 in ys2:
                ax.plot([x1 + node_radius, x2 - node_radius],
                        [y1, y2],
                        color='#757575', linewidth=0.3, alpha=0.4, zorder=0)

    # Input labels
    _, input_ys, input_nodes = all_node_positions[0]
    for y, node_info in zip(input_ys, input_nodes):
        ax.text(layers[0]["x"] - 0.9, y, node_info["label"].split('\n')[0],
                ha='right', va='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#C8E6C9', edgecolor='gray'))

    # Output label
    _, output_ys, _ = all_node_positions[-1]
    ax.text(layers[-1]["x"] + 0.9, output_ys[0], "Output\nLogit",
            ha='left', va='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8BBD9', edgecolor='gray'))

    # Title
    ax.text(6.5, 8, 'EfficientNet-B0 Architecture',
            ha='center', va='center', fontsize=14, weight='bold')

    # Subtitle
    ax.text(6.5, 7.4, 'Pretrained ImageNet → Modified Classifier → Binary Output',
            ha='center', va='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved accurate EfficientNet-B0 diagram to: {output_path}")
    plt.close()


def create_accurate_resnet_diagram(output_path):
    """
    Create accurate ResNet50 diagram showing:
    - Input: 224×224×3
    - Backbone stages with actual channel dimensions
    - Global Average Pooling → 2048 features
    - Classifier: Linear(2048 → 1)
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)

    y_center = 4.5
    node_radius = 0.2

    # Layer configurations with actual ResNet50 dimensions
    layers = [
        {
            "x": 1.5,
            "label": "Input Layer",
            "nodes": [
                {"label": "R\n224×224", "color": "#FF6B6B"},
                {"label": "G\n224×224", "color": "#4ECDC4"},
                {"label": "B\n224×224", "color": "#45B7D1"},
            ],
            "box_color": "#B3E5FC",
            "description": "RGB Image\n224×224×3"
        },
        {
            "x": 4,
            "label": "Initial Conv",
            "nodes": [
                {"label": "64", "color": "#81D4FA"},
            ] * 8,
            "box_color": "#E1BEE7",
            "description": "Conv2d(3→64)\n7×7, Stride 2"
        },
        {
            "x": 6.5,
            "label": "Residual Blocks",
            "nodes": [
                {"label": "256", "color": "#CE93D8"},
                {"label": "512", "color": "#CE93D8"},
                {"label": "1024", "color": "#CE93D8"},
                {"label": "2048", "color": "#CE93D8"},
            ],
            "box_color": "#D1C4E9",
            "description": "Residual Blocks\n(4 stages)"
        },
        {
            "x": 9,
            "label": "Global Avg Pool",
            "nodes": [
                {"label": "2048", "color": "#A5D6A7"},
            ] * 6,  # Show 6 nodes for 2048-dim
            "box_color": "#C8E6C9",
            "description": "Global Avg Pool\n2048 features"
        },
        {
            "x": 11.5,
            "label": "Classifier",
            "nodes": [
                {"label": "1", "color": "#FFCC80"},
            ],
            "box_color": "#FFF9C4",
            "description": "Linear(2048→1)\nTrained"
        },
    ]

    all_node_positions = []

    for layer_idx, layer in enumerate(layers):
        x = layer["x"]
        n_nodes = len(layer["nodes"])
        layer_height = min(5, n_nodes * 0.6)
        y_positions = np.linspace(y_center + layer_height/2, y_center - layer_height/2, n_nodes)
        all_node_positions.append((x, y_positions, layer["nodes"]))

        # Layer label box
        label_box = FancyBboxPatch(
            (x - 0.7, y_center + layer_height/2 + 0.5),
            1.4, 0.8,
            boxstyle="round,pad=0.08",
            facecolor=layer["box_color"],
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(label_box)
        ax.text(x, y_center + layer_height/2 + 0.9, layer["label"],
                ha='center', va='center', fontsize=10, weight='bold')

        # Description below label
        ax.text(x, y_center + layer_height/2 + 0.3, layer["description"],
                ha='center', va='center', fontsize=8, style='italic')

        # Draw nodes
        for i, (y, node_info) in enumerate(zip(y_positions, layer["nodes"])):
            circle = Circle((x, y), node_radius,
                            facecolor=node_info["color"],
                            edgecolor='black',
                            linewidth=1.5)
            ax.add_patch(circle)
            ax.text(x, y, node_info["label"],
                    ha='center', va='center', fontsize=7, weight='bold')

    # Connections between layers
    for layer_idx in range(len(all_node_positions) - 1):
        x1, ys1, nodes1 = all_node_positions[layer_idx]
        x2, ys2, nodes2 = all_node_positions[layer_idx + 1]

        for y1 in ys1:
            for y2 in ys2:
                ax.plot([x1 + node_radius, x2 - node_radius],
                        [y1, y2],
                        color='#757575', linewidth=0.3, alpha=0.4, zorder=0)

    # Input labels
    _, input_ys, input_nodes = all_node_positions[0]
    for y, node_info in zip(input_ys, input_nodes):
        ax.text(layers[0]["x"] - 0.9, y, node_info["label"].split('\n')[0],
                ha='right', va='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#C8E6C9', edgecolor='gray'))

    # Output label
    _, output_ys, _ = all_node_positions[-1]
    ax.text(layers[-1]["x"] + 0.9, output_ys[0], "Output\nLogit",
            ha='left', va='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8BBD9', edgecolor='gray'))

    # Title
    ax.text(6.5, 8, 'ResNet50 Architecture',
            ha='center', va='center', fontsize=14, weight='bold')

    # Subtitle
    ax.text(6.5, 7.4, 'Pretrained ImageNet → Modified Classifier → Binary Output',
            ha='center', va='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved accurate ResNet50 diagram to: {output_path}")
    plt.close()


if __name__ == "__main__":
    from pathlib import Path

    output_dir = Path("Stage1/architecture_diagrams")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating accurate architecture diagrams...")

    create_accurate_efficientnet_diagram(
        output_path=str(output_dir / "efficientnet_b0_accurate.png")
    )

    create_accurate_resnet_diagram(
        output_path=str(output_dir / "resnet50_accurate.png")
    )

    print("\n✓ Accurate diagrams generated!")
    print(f"  Output directory: {output_dir}")
