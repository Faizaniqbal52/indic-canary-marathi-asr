"""
Publication-Grade Empirical Figure Generator for Marathi ASR Technical Report
Generates 5 figures for docs/technical_report.md and PDF compilation.
Strictly adheres to:
1. Zero em dashes in all titles, labels, annotations, and legends.
2. Verified experimental data from EXP-005 and EXP-007.
3. Accurate non-monotonic loss trajectory with minimum at step 170.
4. Speaker-disjoint 20-speaker benchmark framing without calling them generally 'unseen'.
"""

import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure academic plot styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#94A3B8'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#E2E8F0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['figure.facecolor'] = '#FFFFFF'
plt.rcParams['axes.facecolor'] = '#FFFFFF'

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors
COLOR_PRIMARY = '#1E3A8A'     # Navy
COLOR_SECONDARY = '#2563EB'   # Blue
COLOR_ACCENT = '#3B82F6'      # Light Blue
COLOR_GREEN = '#059669'       # Improvement / Green
COLOR_SLATE = '#64748B'       # Invariant / Neutral
COLOR_RED = '#DC2626'         # Degradation / Red
COLOR_TEXT = '#1E293B'        # Dark text
COLOR_MUTED = '#64748B'       # Muted text


def generate_figure_1():
    """Figure 1: Trainable Parameters Across Adaptation Strategies (EXP-005)"""
    fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=300)
    
    strategies = ['Top-2 Layers\n(Strategy A)', 'Top-4 Layers\n(Strategy B)', 'Decoder LoRA\n(r=16, a=32)']
    params_m = [40.93, 74.52, 3.15]
    colors = [COLOR_ACCENT, COLOR_SECONDARY, COLOR_PRIMARY]
    
    bars = ax.bar(strategies, params_m, color=colors, width=0.52, edgecolor='#1E293B', linewidth=0.75, zorder=3)
    ax.grid(axis='y', zorder=0)
    ax.set_axisbelow(True)
    
    # Value labels
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1.8, f"{h:.2f}M",
                ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=COLOR_TEXT)
    
    ax.set_ylabel('Trainable Parameters (Millions)', fontsize=9.5, fontweight='bold', color=COLOR_TEXT)
    ax.set_title('Figure 1: Trainable Parameters Across Adaptation Strategies (EXP-005)',
                 fontsize=10.5, fontweight='bold', pad=14, color=COLOR_TEXT)
    ax.set_ylim(0, 88)
    ax.tick_params(colors=COLOR_TEXT, labelsize=9)
    
    # Callout
    ax.text(2.0, 15.0, "95.8% parameter reduction\nacross all 24 decoder layers",
            ha='center', va='bottom', fontsize=8.0, fontstyle='italic', color=COLOR_PRIMARY,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#EFF6FF', edgecolor='#BFDBFE', lw=0.75))
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig1_trainable_parameters.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Generated: {path}")


def generate_figure_2():
    """Figure 2: Checkpoint Footprint Across Adaptation Strategies (EXP-005)"""
    fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=300)
    
    strategies = ['Top-2 Layers\n(Strategy A)', 'Top-4 Layers\n(Strategy B)', 'Decoder LoRA\n(r=16, a=32)']
    sizes_mb = [156.14, 284.31, 12.07]
    colors = [COLOR_ACCENT, COLOR_SECONDARY, COLOR_PRIMARY]
    
    bars = ax.bar(strategies, sizes_mb, color=colors, width=0.52, edgecolor='#1E293B', linewidth=0.75, zorder=3)
    ax.grid(axis='y', zorder=0)
    ax.set_axisbelow(True)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 6.0, f"{h:.2f} MB",
                ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=COLOR_TEXT)
    
    ax.set_ylabel('Serialized Checkpoint File Size (MB)', fontsize=9.5, fontweight='bold', color=COLOR_TEXT)
    ax.set_title('Figure 2: Checkpoint Footprint Across Adaptation Strategies (EXP-005)',
                 fontsize=10.5, fontweight='bold', pad=14, color=COLOR_TEXT)
    ax.set_ylim(0, 330)
    ax.tick_params(colors=COLOR_TEXT, labelsize=9)
    
    ax.text(2.0, 50.0, "92.3% to 95.8% reduction\nin serialization storage",
            ha='center', va='bottom', fontsize=8.0, fontstyle='italic', color=COLOR_PRIMARY,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#EFF6FF', edgecolor='#BFDBFE', lw=0.75))
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig2_checkpoint_footprint.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Generated: {path}")


def generate_figure_3():
    """Figure 3: Training Loss Trajectory Across 200 Optimization Steps (EXP-007)"""
    curves_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'google_drive_submission', '02_training_logs', 'training_curves.json')
    with open(curves_path, 'r', encoding='utf-8') as f:
        curves = json.load(f)
        
    steps = [x['step'] for x in curves]
    losses = [x['loss'] for x in curves]
    
    fig, ax = plt.subplots(figsize=(6.5, 3.4), dpi=300)
    
    ax.plot(steps, losses, color=COLOR_SECONDARY, linewidth=2.0, marker='o', markersize=4.5,
            markerfacecolor=COLOR_PRIMARY, markeredgecolor='#FFFFFF', markeredgewidth=0.75, zorder=3,
            label='Cross-Entropy Loss (FP32)')
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)
    
    # Annotate key points
    # Step 1
    ax.annotate(f"Step 1: {losses[0]:.4f}", xy=(steps[0], losses[0]), xytext=(steps[0] + 12, losses[0] + 0.15),
                fontsize=8.5, fontweight='bold', color=COLOR_TEXT,
                arrowprops=dict(arrowstyle='->', color='#475569', lw=0.8))
    
    # Step 170 (Minimum)
    min_idx = 17  # Step 170 is 1.6160
    ax.annotate(f"Min Loss (Step 170): {losses[min_idx]:.4f}", xy=(steps[min_idx], losses[min_idx]),
                xytext=(steps[min_idx] - 65, losses[min_idx] - 0.28),
                fontsize=8.5, fontweight='bold', color=COLOR_GREEN,
                arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=1.0))
    
    # Step 200 (Final)
    ax.annotate(f"Final Step 200: {losses[-1]:.4f}\n(-45.60% relative)", xy=(steps[-1], losses[-1]),
                xytext=(steps[-1] - 48, losses[-1] + 0.35),
                fontsize=8.5, fontweight='bold', color=COLOR_PRIMARY,
                arrowprops=dict(arrowstyle='->', color=COLOR_PRIMARY, lw=1.0))
    
    ax.set_xlabel('Optimization Step (Batch Size = 1, Gradient Accumulation = 4)', fontsize=9.5, fontweight='bold', color=COLOR_TEXT)
    ax.set_ylabel('Training Cross-Entropy Loss', fontsize=9.5, fontweight='bold', color=COLOR_TEXT)
    ax.set_title('Figure 3: Training Loss Trajectory Across 200 Optimization Steps (EXP-007)',
                 fontsize=10.5, fontweight='bold', pad=14, color=COLOR_TEXT)
    ax.set_xlim(-5, 215)
    ax.set_ylim(1.2, 3.4)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.tick_params(colors=COLOR_TEXT, labelsize=9)
    
    # Explanatory note
    ax.text(0.03, 0.10, "Non-monotonic convergence trajectory: overall reduction of 45.60%,\nminimum at step 170 (1.6160), slight inflection to 1.6670 at step 200.",
            transform=ax.transAxes, fontsize=8.0, fontstyle='italic', color=COLOR_MUTED,
            bbox=dict(boxstyle='square,pad=0.4', facecolor='#F8FAFC', edgecolor='#E2E8F0', lw=0.5))
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig3_training_loss_curve.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Generated: {path}")


def generate_figure_4():
    """Figure 4: Held-Out Marathi ASR Base vs Fine-Tuned WER (EXP-007)"""
    fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=300)
    
    models = ['Zero-Shot Base Model', 'LoRA Fine-Tuned (200 Steps)']
    wer_vals = [29.40, 31.29]
    colors = [COLOR_SECONDARY, '#475569']
    
    bars = ax.bar(models, wer_vals, color=colors, width=0.42, edgecolor='#1E293B', linewidth=0.75, zorder=3)
    ax.grid(axis='y', zorder=0)
    ax.set_axisbelow(True)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.8, f"{h:.2f}%",
                ha='center', va='bottom', fontsize=10.0, fontweight='bold', color=COLOR_TEXT)
        
    ax.set_ylabel('Word Error Rate (%)', fontsize=9.5, fontweight='bold', color=COLOR_TEXT)
    ax.set_title('Figure 4: Held-Out Marathi ASR: Base vs. Fine-Tuned WER (EXP-007)',
                 fontsize=10.5, fontweight='bold', pad=14, color=COLOR_TEXT)
    ax.set_ylim(0, 38)
    ax.tick_params(colors=COLOR_TEXT, labelsize=9)
    
    # Delta indication
    ax.annotate("", xy=(1.0, 32.5), xytext=(0.0, 30.5),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.25", color=COLOR_RED, lw=1.2))
    ax.text(0.5, 33.5, "+1.89 pp (Aggregate Drift)", ha='center', va='bottom',
            fontsize=8.5, fontweight='bold', color=COLOR_RED)
    
    # Footnote
    ax.text(0.5, 0.08, "Evaluated on frozen 100-utterance benchmark under identical greedy decoding.\nAggregate WER divergence motivated speaker and linguistic error stratification.",
            transform=ax.transAxes, ha='center', fontsize=7.8, fontstyle='italic', color=COLOR_MUTED)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig4_base_vs_finetuned_wer.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Generated: {path}")


def generate_figure_5():
    """Figure 5: Per-Speaker WER Change on the 20-Speaker Evaluation Benchmark"""
    eval_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'google_drive_submission', '03_evaluation_predictions', 'final_evaluation_results.json')
    with open(eval_path, 'r', encoding='utf-8') as f:
        ev = json.load(f)
        
    speakers = defaultdict(lambda: {'base': [], 'ft': [], 'gender': None})
    for item in ev['detailed_results']:
        spk = str(item.get('speaker_id'))
        gender = item.get('gender')
        base_wer = item.get('wer_base_norm', item.get('wer_base', 0.0))
        ft_wer = item.get('wer_finetuned_norm', item.get('wer_finetuned', 0.0))
        speakers[spk]['base'].append(base_wer)
        speakers[spk]['ft'].append(ft_wer)
        speakers[spk]['gender'] = gender
        
    spk_data = []
    for spk, d in speakers.items():
        base_avg = sum(d['base']) / len(d['base']) * 100
        ft_avg = sum(d['ft']) / len(d['ft']) * 100
        delta = ft_avg - base_avg
        spk_data.append({
            'spk': spk,
            'label': f"Spk {spk} ({d['gender'][0]})",
            'delta': delta,
            'base': base_avg,
            'ft': ft_avg
        })
        
    # Sort from largest improvement to largest regression
    spk_data.sort(key=lambda x: x['delta'], reverse=True)
    
    labels = [x['label'] for x in spk_data]
    deltas = [x['delta'] for x in spk_data]
    
    # Colors: Green for improvement (< -0.1), Slate for invariant (-0.1 to 0.1), Red for regression (> 0.1)
    bar_colors = []
    for d in deltas:
        if d < -0.1:
            bar_colors.append(COLOR_GREEN)
        elif abs(d) <= 0.1:
            bar_colors.append(COLOR_SLATE)
        else:
            bar_colors.append(COLOR_RED)
            
    fig, ax = plt.subplots(figsize=(6.8, 5.0), dpi=300)
    
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, deltas, color=bar_colors, height=0.68, edgecolor='#1E293B', linewidth=0.6, zorder=3)
    ax.grid(axis='x', zorder=0)
    ax.set_axisbelow(True)
    
    # Reference zero line
    ax.axvline(0, color='#1E293B', linewidth=0.9, zorder=4)
    
    # Numerical labels beside bars
    for idx, (bar, d) in enumerate(zip(bars, deltas)):
        if d < 0:
            ax.text(d - 0.35, bar.get_y() + bar.get_height() / 2.0, f"{d:.2f} pp",
                    ha='right', va='center', fontsize=7.5, fontweight='bold', color=COLOR_GREEN)
        elif d == 0:
            ax.text(0.35, bar.get_y() + bar.get_height() / 2.0, "0.00 pp",
                    ha='left', va='center', fontsize=7.5, color=COLOR_SLATE)
        else:
            ax.text(d + 0.35, bar.get_y() + bar.get_height() / 2.0, f"+{d:.2f} pp",
                    ha='left', va='center', fontsize=7.5, fontweight='bold', color=COLOR_RED)
            
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.0, color=COLOR_TEXT)
    ax.set_xlabel('Absolute Word Error Rate Delta (Percentage Points, Fine-Tuned minus Base)',
                  fontsize=8.8, fontweight='bold', color=COLOR_TEXT)
    ax.set_title('Figure 5: Per-Speaker WER Change on the 20-Speaker Evaluation Benchmark (EXP-007)',
                 fontsize=10.0, fontweight='bold', pad=12, color=COLOR_TEXT)
    ax.set_xlim(-13, 14)
    ax.tick_params(colors=COLOR_TEXT)
    
    # Direction callouts
    ax.text(-8.0, 1.2, "<-- Acoustic Improvement (Lower WER)", fontsize=7.5, fontweight='bold', color=COLOR_GREEN)
    ax.text(2.5, 18.2, "Acoustic Drift / Degradation (Higher WER) -->", fontsize=7.5, fontweight='bold', color=COLOR_RED)
    
    # Benchmark note
    ax.text(0.02, 0.02, "Benchmark is strictly speaker-disjoint from derived fine-tuning partition (5 samples/speaker).\nIncludes all 20 benchmark speakers without selective exclusion (10 Female, 10 Male).",
            transform=ax.transAxes, fontsize=7.2, fontstyle='italic', color=COLOR_MUTED,
            bbox=dict(boxstyle='square,pad=0.3', facecolor='#F8FAFC', edgecolor='#E2E8F0', lw=0.5))
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig5_per_speaker_wer_delta.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Generated: {path}")


if __name__ == '__main__':
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    generate_figure_4()
    generate_figure_5()
    print("All 5 figures generated successfully.")
