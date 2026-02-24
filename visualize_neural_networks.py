"""
Generate neural network architecture diagrams.
- Classic style: circular nodes, layer labels above, full connectivity (like the reference image).
- Detailed style: layer boxes with more technical detail.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np


def create_classic_nn_diagram(
    model_name,
    backbone_name,
    feature_dim,
    output_path,
    n_input=3,
    n_hidden_backbone=5,
    n_hidden_classifier=4,
    n_output=1,
):
    """
    Create a classic feedforward neural network diagram with accurate dimensions:
    - Circular nodes in columns
    - Layer labels in boxes above each column showing actual dimensions
    - Full connectivity (every node connected to every node in next layer)
    - Input labels on the left, output labels on the right
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.axis('off')
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10)  # Increased height to accommodate title

    # Title at the top (before drawing layers)
    ax.text(6.5, 9.2, f'{model_name} Neural Network',
            ha='center', va='center', fontsize=14, weight='bold')
    
    # Subtitle below title
    ax.text(6.5, 8.6, f'Pretrained ImageNet → Modified Classifier → Binary Output',
            ha='center', va='center', fontsize=10, style='italic')

    # Layer positions (x) and colors with accurate labels
    layer_configs = [
        {
            "x": 2, 
            "n_nodes": n_input, 
            "label": "Input Layer\n224×224×3", 
            "color": "#B3E5FC", 
            "node_color": "#81D4FA",
            "detail": "RGB Channels\n(150,528 values)"
        },
        {
            "x": 5, 
            "n_nodes": n_hidden_backbone, 
            "label": f"Pretrained\n{backbone_name}\nBackbone", 
            "color": "#E1BEE7", 
            "node_color": "#CE93D8",
            "detail": f"{feature_dim} features\n(shown: {n_hidden_backbone} nodes)"
        },
        {
            "x": 8, 
            "n_nodes": n_hidden_classifier, 
            "label": f"Classifier\nLinear({feature_dim}→1)", 
            "color": "#D1C4E9", 
            "node_color": "#B39DDB",
            "detail": f"{feature_dim} inputs\n(shown: {n_hidden_classifier} nodes)"
        },
        {
            "x": 11, 
            "n_nodes": n_output, 
            "label": "Output Layer\nBinary", 
            "color": "#FFE0B2", 
            "node_color": "#FFCC80",
            "detail": "Sigmoid → Prob"
        },
    ]

    y_center = 4.5  # Lowered to make room for title
    node_radius = 0.22
    layer_height = 3.0  # vertical span for nodes

    all_positions = []  # list of (x, [y_positions]) per layer

    for cfg in layer_configs:
        n = cfg["n_nodes"]
        x = cfg["x"]
        y_positions = np.linspace(y_center + layer_height/2, y_center - layer_height/2, n)
        all_positions.append((x, y_positions))

        # Layer label box above column (positioned lower to avoid title)
        label_y_top = y_center + layer_height/2 + 0.3
        label_box = FancyBboxPatch(
            (x - 0.75, label_y_top),
            1.5, 1.0,
            boxstyle="round,pad=0.08",
            facecolor=cfg["color"],
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(label_box)
        ax.text(x, label_y_top + 0.7, cfg["label"],
                ha='center', va='center', fontsize=9, weight='bold')
        ax.text(x, label_y_top + 0.15, cfg["detail"],
                ha='center', va='center', fontsize=7, style='italic')

        # Nodes (circles)
        for y in y_positions:
            circle = Circle((x, y), node_radius,
                            facecolor=cfg["node_color"],
                            edgecolor='black',
                            linewidth=1.5)
            ax.add_patch(circle)

    # Input labels (left of input nodes)
    _, input_ys = all_positions[0]
    input_labels = ["R\n224×224", "G\n224×224", "B\n224×224"]
    for i, y in enumerate(input_ys):
        lbl = input_labels[i] if i < len(input_labels) else f"Input #{i+1}"
        ax.text(layer_configs[0]["x"] - 0.9, y, lbl,
                ha='right', va='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#C8E6C9', edgecolor='gray'))

    # Output labels (right of output nodes)
    _, output_ys = all_positions[-1]
    for i, y in enumerate(output_ys):
        ax.text(layer_configs[-1]["x"] + 0.9, y, "Output\nLogit",
                ha='left', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#F8BBD9', edgecolor='gray'))

    # Connections: every node in layer i to every node in layer i+1
    for layer_idx in range(len(all_positions) - 1):
        x1, ys1 = all_positions[layer_idx]
        x2, ys2 = all_positions[layer_idx + 1]
        for y1 in ys1:
            for y2 in ys2:
                ax.plot([x1 + node_radius, x2 - node_radius],
                        [y1, y2],
                        color='#424242', linewidth=0.4, zorder=0)

    # Note about representation (at the bottom)
    note_text = f"Note: Nodes shown are representative. Actual dimensions:\n"
    note_text += f"Input: 224×224×3 = 150,528 values | Backbone: {feature_dim} features | Classifier: {feature_dim}→1"
    ax.text(6.5, 0.5, note_text,
            ha='center', va='center', fontsize=7, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFDE7', edgecolor='gray', alpha=0.8))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved classic NN diagram to: {output_path}")
    plt.close()


def create_classic_nn_diagram_resnet(
    model_name,
    backbone_name,
    feature_dim,
    output_path,
    n_input=3,
    n_hidden_backbone=6,
    n_hidden_classifier=4,
    n_output=1,
):
    """Same as above but with ResNet50 (2048-dim) - show 6 nodes in backbone column."""
    create_classic_nn_diagram(
        model_name=model_name,
        backbone_name=backbone_name,
        feature_dim=feature_dim,
        output_path=output_path,
        n_input=n_input,
        n_hidden_backbone=n_hidden_backbone,
        n_hidden_classifier=n_hidden_classifier,
        n_output=n_output,
    )


def create_neural_network_diagram(model_name, backbone_name, feature_dim, num_blocks=5, output_path=None):
    """
    Create a neural network diagram showing the actual network structure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    input_color = '#E1F5FE'
    conv_color = '#BBDEFB'
    pool_color = '#90CAF9'
    fc_color = '#C8E6C9'
    output_color = '#FFF9C4'

    y_center = 5
    x_pos = 2.5
    block_width = 1.2
    block_height = 2.5
    block_spacing = 1.5

    # Input
    input_box = FancyBboxPatch(
        (0.5, y_center - 1.5), 1.2, 3,
        boxstyle="round,pad=0.1",
        facecolor=input_color,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(input_box)
    ax.text(1.1, y_center + 1, 'Input Image', ha='center', va='center', fontsize=11, weight='bold')
    ax.text(1.1, y_center, '224×224×3', ha='center', va='center', fontsize=10, family='monospace')
    ax.text(1.1, y_center - 1, 'RGB', ha='center', va='center', fontsize=9)

    ax.text(7, 8.5, f'Pretrained {backbone_name} Backbone',
            ha='center', va='center', fontsize=12, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', edgecolor='black', linewidth=2))

    for i in range(num_blocks):
        x_block = x_pos + i * block_spacing
        block = FancyBboxPatch(
            (x_block - block_width/2, y_center - block_height/2),
            block_width, block_height,
            boxstyle="round,pad=0.1",
            facecolor=conv_color,
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(block)
        if i == 0:
            ax.text(x_block, y_center + 0.8, 'Conv', ha='center', va='center', fontsize=9, weight='bold')
            ax.text(x_block, y_center, 'Block 1', ha='center', va='center', fontsize=8)
        elif i == num_blocks - 1:
            ax.text(x_block, y_center + 0.8, 'Conv', ha='center', va='center', fontsize=9, weight='bold')
            ax.text(x_block, y_center, f'Block {num_blocks}', ha='center', va='center', fontsize=8)
            ax.text(x_block, y_center - 0.8, f'→ {feature_dim}D', ha='center', va='center', fontsize=8, style='italic')
        else:
            ax.text(x_block, y_center + 0.8, 'Conv', ha='center', va='center', fontsize=9, weight='bold')
            ax.text(x_block, y_center, f'Block {i+1}', ha='center', va='center', fontsize=8)
        if i < num_blocks - 1:
            arrow = FancyArrowPatch(
                (x_block + block_width/2, y_center),
                (x_block + block_spacing - block_width/2, y_center),
                arrowstyle='->', mutation_scale=15, linewidth=1.5,
                color='black', zorder=3
            )
            ax.add_patch(arrow)

    arrow_input = FancyArrowPatch(
        (1.7, y_center),
        (x_pos - block_width/2, y_center),
        arrowstyle='->', mutation_scale=20, linewidth=2,
        color='black', zorder=3
    )
    ax.add_patch(arrow_input)

    x_features = x_pos + (num_blocks - 1) * block_spacing + 1.5
    features_box = FancyBboxPatch(
        (x_features - 0.8, y_center - 1),
        1.6, 2,
        boxstyle="round,pad=0.1",
        facecolor=pool_color,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(features_box)
    ax.text(x_features, y_center + 0.3, 'Global', ha='center', va='center', fontsize=9, weight='bold')
    ax.text(x_features, y_center - 0.2, 'Avg Pool', ha='center', va='center', fontsize=9, weight='bold')
    ax.text(x_features, y_center - 0.8, f'{feature_dim}-dim', ha='center', va='center', fontsize=8, style='italic')

    arrow_features = FancyArrowPatch(
        (x_pos + (num_blocks - 1) * block_spacing + block_width/2, y_center),
        (x_features - 0.8, y_center),
        arrowstyle='->', mutation_scale=20, linewidth=2,
        color='black', zorder=3
    )
    ax.add_patch(arrow_features)

    x_classifier = x_features + 2.5
    classifier_box = FancyBboxPatch(
        (x_classifier - 1, y_center - 1.2),
        2, 2.4,
        boxstyle="round,pad=0.1",
        facecolor=fc_color,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(classifier_box)
    ax.text(x_classifier, y_center + 0.9, 'Modified Classifier',
            ha='center', va='center', fontsize=10, weight='bold')
    ax.text(x_classifier, y_center + 0.5, f'Linear({feature_dim} → 1)',
            ha='center', va='center', fontsize=9, family='monospace')
    ax.text(x_classifier, y_center - 1.0, 'Trained from scratch',
            ha='center', va='center', fontsize=8, style='italic')

    arrow_classifier = FancyArrowPatch(
        (x_features + 0.8, y_center),
        (x_classifier - 1, y_center),
        arrowstyle='->', mutation_scale=20, linewidth=2,
        color='black', zorder=3
    )
    ax.add_patch(arrow_classifier)

    x_output = x_classifier + 2.5
    output_box = FancyBboxPatch(
        (x_output - 0.6, y_center - 0.8),
        1.2, 1.6,
        boxstyle="round,pad=0.1",
        facecolor=output_color,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(output_box)
    ax.text(x_output, y_center - 0.1, 'Logit', ha='center', va='center', fontsize=9, weight='bold')
    ax.text(x_output, y_center - 0.6, 'Sigmoid', ha='center', va='center', fontsize=8)
    ax.text(x_output, y_center - 0.9, '→ Prob', ha='center', va='center', fontsize=8, style='italic')

    arrow_output = FancyArrowPatch(
        (x_classifier + 1, y_center - 0.5),
        (x_output - 0.6, y_center - 0.1),
        arrowstyle='->', mutation_scale=20, linewidth=2,
        color='black', zorder=3
    )
    ax.add_patch(arrow_output)

    ax.text(7, 9.2, f'{model_name} Neural Network Architecture',
            ha='center', va='center', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved neural network diagram to: {output_path}")
    plt.close()


def create_detailed_network_diagram(model_name, backbone_name, feature_dim, output_path):
    """Create a more detailed network diagram with explicit layer structure."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis('off')
    y_center = 5.5
    x_start = 2.5
    layer_width = 1.8
    layer_height = 3.5
    layer_spacing = 2.0
    layers = [
        ('Conv2D', '3→32', '#BBDEFB'),
        ('Conv2D', '32→64', '#90CAF9'),
        ('Conv2D', '64→128', '#64B5F6'),
        ('Conv2D', '128→256', '#42A5F5'),
        ('Conv2D', f'256→{feature_dim}', '#2196F3'),
    ]
    input_layer = FancyBboxPatch(
        (0.3, y_center - 2), 1.4, 4,
        boxstyle="round,pad=0.15",
        facecolor='#E1F5FE',
        edgecolor='black',
        linewidth=2.5
    )
    ax.add_patch(input_layer)
    ax.text(1.0, y_center + 1.2, 'INPUT', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(1.0, y_center, '224×224×3', ha='center', va='center', fontsize=11, family='monospace')
    ax.text(1.0, y_center - 1.2, 'RGB Image', ha='center', va='center', fontsize=10)
    ax.text(8, 9.5, f'PRETRAINED {backbone_name.upper()} BACKBONE',
            ha='center', va='center', fontsize=13, weight='bold',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#E3F2FD', edgecolor='black', linewidth=2.5))
    for i, (layer_name, channels, color) in enumerate(layers):
        x_layer = x_start + i * layer_spacing
        layer_box = FancyBboxPatch(
            (x_layer - layer_width/2, y_center - layer_height/2),
            layer_width, layer_height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(layer_box)
        ax.text(x_layer, y_center + 1.2, layer_name, ha='center', va='center', fontsize=10, weight='bold')
        ax.text(x_layer, y_center + 0.3, channels, ha='center', va='center', fontsize=9, family='monospace')
        if i < len(layers) - 1:
            arrow = FancyArrowPatch(
                (x_layer + layer_width/2, y_center),
                (x_layer + layer_spacing - layer_width/2, y_center),
                arrowstyle='->', mutation_scale=25, linewidth=2.5,
                color='black', zorder=3
            )
            ax.add_patch(arrow)
    x_pool = x_start + len(layers) * layer_spacing
    pool_box = FancyBboxPatch(
        (x_pool - 0.9, y_center - 1.2),
        1.8, 2.4,
        boxstyle="round,pad=0.1",
        facecolor='#81C784',
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(pool_box)
    ax.text(x_pool, y_center + 0.4, 'Global', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(x_pool, y_center - 0.2, 'Avg Pool', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(x_pool, y_center - 0.8, f'{feature_dim}D', ha='center', va='center', fontsize=9, family='monospace', weight='bold')
    arrow_pool = FancyArrowPatch(
        (x_start + (len(layers) - 1) * layer_spacing + layer_width/2, y_center),
        (x_pool - 0.9, y_center),
        arrowstyle='->', mutation_scale=25, linewidth=2.5,
        color='black', zorder=3
    )
    ax.add_patch(arrow_pool)
    x_fc = x_pool + 2.2
    fc_box = FancyBboxPatch(
        (x_fc - 1.2, y_center - 1.8),
        2.4, 3.6,
        boxstyle="round,pad=0.15",
        facecolor='#C8E6C9',
        edgecolor='black',
        linewidth=2.5
    )
    ax.add_patch(fc_box)
    ax.text(x_fc, y_center + 1.2, 'MODIFIED', ha='center', va='center', fontsize=11, weight='bold')
    ax.text(x_fc, y_center + 0.6, 'CLASSIFIER', ha='center', va='center', fontsize=11, weight='bold')
    ax.text(x_fc, y_center, f'Linear({feature_dim} → 1)', ha='center', va='center', fontsize=10, family='monospace')
    ax.text(x_fc, y_center - 1.4, 'Trained from scratch', ha='center', va='center', fontsize=9, style='italic', weight='bold')
    arrow_fc = FancyArrowPatch(
        (x_pool + 0.9, y_center),
        (x_fc - 1.2, y_center),
        arrowstyle='->', mutation_scale=25, linewidth=2.5,
        color='black', zorder=3
    )
    ax.add_patch(arrow_fc)
    x_out = x_fc + 2.5
    out_box = FancyBboxPatch(
        (x_out - 0.7, y_center - 1),
        1.4, 2,
        boxstyle="round,pad=0.15",
        facecolor='#FFF9C4',
        edgecolor='black',
        linewidth=2.5
    )
    ax.add_patch(out_box)
    out_neuron = Circle((x_out, y_center - 0.2), 0.18,
                        facecolor='white', edgecolor='black', linewidth=2.5)
    ax.add_patch(out_neuron)
    ax.text(x_out, y_center - 0.2, 'Logit', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(x_out, y_center - 0.7, 'Sigmoid', ha='center', va='center', fontsize=9)
    ax.text(x_out, y_center - 1.0, '→ Prob', ha='center', va='center', fontsize=8, style='italic')
    arrow_out = FancyArrowPatch(
        (x_fc + 1.2, y_center - 0.4),
        (x_out - 0.7, y_center - 0.2),
        arrowstyle='->', mutation_scale=25, linewidth=2.5,
        color='black', zorder=3
    )
    ax.add_patch(arrow_out)
    ax.text(8, 10.5, f'{model_name} Neural Network Architecture',
            ha='center', va='center', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved detailed neural network diagram to: {output_path}")
    plt.close()


if __name__ == "__main__":
    from pathlib import Path

    output_dir = Path("Stage1/architecture_diagrams")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating neural network architecture diagrams...")

    # --- Classic style (like the reference image) ---
    # Note: Input shows 3 nodes (RGB channels) - actual input is 224×224×3 = 150,528 values
    # Backbone shows representative nodes - actual EfficientNet-B0 produces 1280 features
    create_classic_nn_diagram(
        model_name="EfficientNet-B0",
        backbone_name="EfficientNet-B0",
        feature_dim=1280,
        output_path=str(output_dir / "efficientnet_b0_classic.png"),
        n_input=3,  # RGB channels (represents 224×224×3 image)
        n_hidden_backbone=12,  # Representative of 1280 features
        n_hidden_classifier=8,  # Representative of 1280→1 transformation
        n_output=1,
    )

    # Backbone shows representative nodes - actual ResNet50 produces 2048 features
    create_classic_nn_diagram(
        model_name="ResNet50",
        backbone_name="ResNet50",
        feature_dim=2048,
        output_path=str(output_dir / "resnet50_classic.png"),
        n_input=3,  # RGB channels (represents 224×224×3 image)
        n_hidden_backbone=15,  # Representative of 2048 features
        n_hidden_classifier=10,  # Representative of 2048→1 transformation
        n_output=1,
    )

    # --- Previous styles (optional) ---
    create_neural_network_diagram(
        model_name="EfficientNet-B0",
        backbone_name="EfficientNet-B0",
        feature_dim=1280,
        num_blocks=5,
        output_path=str(output_dir / "efficientnet_b0_neural_network.png")
    )

    create_neural_network_diagram(
        model_name="ResNet50",
        backbone_name="ResNet50",
        feature_dim=2048,
        num_blocks=5,
        output_path=str(output_dir / "resnet50_neural_network.png")
    )

    create_detailed_network_diagram(
        model_name="EfficientNet-B0",
        backbone_name="EfficientNet-B0",
        feature_dim=1280,
        output_path=str(output_dir / "efficientnet_b0_detailed.png")
    )

    create_detailed_network_diagram(
        model_name="ResNet50",
        backbone_name="ResNet50",
        feature_dim=2048,
        output_path=str(output_dir / "resnet50_detailed.png")
    )

    print("\n✓ All diagrams generated!")
    print(f"  Classic style (like your reference): efficientnet_b0_classic.png, resnet50_classic.png")
    print(f"  Output directory: {output_dir}")
