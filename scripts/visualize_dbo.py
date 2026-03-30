#!/usr/bin/env python3
"""
Visualize DBO (Dual Batch Overlap) pipeline timing.

Reads timing JSON files and generates timeline visualization showing
how computation and communication overlap.

Usage:
    python scripts/visualize_dbo.py results/timing_attention.json results/timing_ffn.json
    python scripts/visualize_dbo.py results/  # auto-detect files
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
except ImportError:
    print("Error: matplotlib required. Install with: pip install matplotlib")
    sys.exit(1)


# Color scheme for different event types
COLORS = {
    "attn_compute": "#4CAF50",   # Green
    "ffn_compute": "#2196F3",    # Blue  
    "moe_router": "#E91E63",     # Pink
    "moe_experts": "#00BCD4",    # Cyan
    "moe_shared_or_dense": "#795548",  # Brown
    "send_wait": "#FF9800",      # Orange
    "recv_wait": "#9C27B0",      # Purple
}

LABELS = {
    "attn_compute": "Attention",
    "ffn_compute": "FFN",
    "moe_router": "MoE Router",
    "moe_experts": "MoE Experts",
    "moe_shared_or_dense": "MoE Shared/Dense",
    "send_wait": "Send",
    "recv_wait": "Recv",
}


def load_timing(path: str) -> dict:
    """Load timing data from JSON file."""
    with open(path) as f:
        return json.load(f)


def plot_timeline(attn_data: dict, ffn_data: dict, output_path: str, 
                  max_layers: int = 5):
    """
    Plot timeline showing DBO overlap.
    
    Args:
        attn_data: Timing data from attention node
        ffn_data: Timing data from FFN node
        output_path: Path to save the figure
        max_layers: Max layers to show (for readability)
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    num_layers = min(attn_data.get("num_layers", 1), max_layers)
    num_mb = attn_data.get("num_micro_batches", 2)
    
    # Plot attention node
    ax_attn = axes[0]
    ax_attn.set_title(f"Attention Node (layers 0-{num_layers-1})", fontsize=12)
    _plot_node_timeline(ax_attn, attn_data, num_layers, num_mb)
    
    # Plot FFN node
    ax_ffn = axes[1]
    ax_ffn.set_title(f"FFN Node (layers 0-{num_layers-1})", fontsize=12)
    _plot_node_timeline(ax_ffn, ffn_data, num_layers, num_mb)
    
    # Common x-axis label
    ax_ffn.set_xlabel("Time (ms)", fontsize=11)
    
    # Legend
    handles = [mpatches.Patch(color=color, label=LABELS[evt]) 
               for evt, color in COLORS.items()]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 0.98))
    
    # Add summary stats
    attn_total = attn_data.get("total_time_ms", 0)
    ffn_total = ffn_data.get("total_time_ms", 0)
    attn_ratio = attn_data.get("compute_ratio", 0) * 100
    ffn_ratio = ffn_data.get("compute_ratio", 0) * 100
    
    summary = (f"E2E: Attn={attn_total:.1f}ms, FFN={ffn_total:.1f}ms | "
               f"Compute Ratio: Attn={attn_ratio:.1f}%, FFN={ffn_ratio:.1f}%")
    fig.suptitle(f"DBO Pipeline Timeline (relative to each node)\n{summary}", fontsize=13, y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    # Also show
    plt.show()


def _plot_node_timeline(ax, data: dict, max_layers: int, num_mb: int):
    """Plot timeline for a single node."""
    events = data.get("events", [])
    
    if not events:
        ax.text(0.5, 0.5, "No timing data", ha='center', va='center', 
                transform=ax.transAxes)
        return
    
    # Y-axis: one row per (layer, mb) combination
    y_positions = {}
    y_idx = 0
    for layer in range(max_layers):
        for mb in range(num_mb):
            y_positions[(layer, mb)] = y_idx
            y_idx += 1
    
    y_labels = []
    for layer in range(max_layers):
        for mb in range(num_mb):
            y_labels.append(f"L{layer} MB{mb}")
    
    # Plot events as rectangles
    for event in events:
        layer = event.get("layer", 0)
        mb = event.get("mb", 0)
        
        if layer >= max_layers:
            continue
        
        evt_type = event.get("type", "")
        start = event.get("start", 0) * 1000  # Convert to ms
        end = event.get("end", 0) * 1000
        duration = end - start
        
        y = y_positions.get((layer, mb), 0)
        color = COLORS.get(evt_type, "#888888")
        
        rect = Rectangle((start, y - 0.4), duration, 0.8,
                         facecolor=color, edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
    
    # Configure axes
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_ylabel("Layer / MicroBatch")
    ax.set_ylim(-0.5, len(y_labels) - 0.5)
    
    # Auto x-limit based on events
    if events:
        max_time = max(e.get("end", 0) * 1000 for e in events if e.get("layer", 0) < max_layers)
        ax.set_xlim(0, max_time * 1.05)
    
    ax.grid(True, axis='x', alpha=0.3)


def plot_overlap_analysis(attn_data: dict, ffn_data: dict, output_path: str):
    """
    Plot analysis of computation vs waiting time (corrected metrics).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Time breakdown (compute vs recv_wait only)
    ax1 = axes[0]
    categories = ['Attention\nCompute', 'FFN\nCompute', 'Attention\nWait', 'FFN\nWait']
    values = [
        attn_data.get("total_compute_ms", 0),
        ffn_data.get("total_compute_ms", 0),
        attn_data.get("total_recv_wait_ms", 0),
        ffn_data.get("total_recv_wait_ms", 0),
    ]
    colors = [COLORS["attn_compute"], COLORS["ffn_compute"], 
              COLORS["recv_wait"], COLORS["recv_wait"]]
    
    bars = ax1.bar(categories, values, color=colors)
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Compute vs Wait Time")
    
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # Add MoE breakdown text if available
    attn_moe = (
        attn_data.get("total_moe_router_ms", 0),
        attn_data.get("total_moe_experts_ms", 0),
        attn_data.get("total_moe_shared_or_dense_ms", 0),
    )
    ffn_moe = (
        ffn_data.get("total_moe_router_ms", 0),
        ffn_data.get("total_moe_experts_ms", 0),
        ffn_data.get("total_moe_shared_or_dense_ms", 0),
    )
    if sum(attn_moe) > 0 or sum(ffn_moe) > 0:
        ax1.text(
            0.02, 0.98,
            (
                f"Attn MoE: router={attn_moe[0]:.1f}ms, experts={attn_moe[1]:.1f}ms, shared/dense={attn_moe[2]:.1f}ms\n"
                f"FFN MoE: router={ffn_moe[0]:.1f}ms, experts={ffn_moe[1]:.1f}ms, shared/dense={ffn_moe[2]:.1f}ms"
            ),
            transform=ax1.transAxes,
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
        )
    
    # Right: Compute ratio (DBO efficiency)
    ax2 = axes[1]
    
    attn_ratio = attn_data.get("compute_ratio", 0) * 100
    ffn_ratio = ffn_data.get("compute_ratio", 0) * 100
    
    nodes = ['Attention', 'FFN']
    ratios = [attn_ratio, ffn_ratio]
    colors = [COLORS["attn_compute"], COLORS["ffn_compute"]]
    
    bars = ax2.bar(nodes, ratios, color=colors)
    ax2.set_ylabel("Compute Ratio (%)")
    ax2.set_title("DBO Efficiency (compute_time / e2e_time)")
    ax2.set_ylim(0, 100)
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
    
    for bar, val in zip(bars, ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize DBO timing")
    parser.add_argument("paths", nargs="*", default=["results"],
                       help="Timing JSON files or directory")
    parser.add_argument("--output", "-o", default="results/dbo_timeline.png",
                       help="Output image path")
    parser.add_argument("--max-layers", type=int, default=5,
                       help="Max layers to show")
    args = parser.parse_args()
    
    # Find timing files
    attn_file = None
    ffn_file = None
    
    for path in args.paths:
        if os.path.isdir(path):
            for f in os.listdir(path):
                if "attention" in f and f.endswith(".json"):
                    attn_file = os.path.join(path, f)
                elif "ffn" in f and f.endswith(".json"):
                    ffn_file = os.path.join(path, f)
        elif "attention" in path:
            attn_file = path
        elif "ffn" in path:
            ffn_file = path
    
    if not attn_file or not ffn_file:
        print("Error: Need both attention and FFN timing files")
        print("Run with --timing flag first:")
        print("  torchrun ... --scheduler async --timing")
        sys.exit(1)
    
    print(f"Loading: {attn_file}")
    print(f"Loading: {ffn_file}")
    
    attn_data = load_timing(attn_file)
    ffn_data = load_timing(ffn_file)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    # Generate plots
    plot_timeline(attn_data, ffn_data, args.output, args.max_layers)
    
    # Also generate analysis plot
    analysis_path = args.output.replace(".png", "_analysis.png")
    plot_overlap_analysis(attn_data, ffn_data, analysis_path)
    
    print("\nDone! Generated visualizations:")
    print(f"  - {args.output}")
    print(f"  - {analysis_path}")


if __name__ == "__main__":
    main()
