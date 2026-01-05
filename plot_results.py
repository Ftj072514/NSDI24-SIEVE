#!/usr/bin/env python3
"""
SIEVE Cache Algorithm - Visualization Script
=============================================
Generates publication-quality figures:
- Figure A: Hit Rate vs Cache Size
- Figure B: Impact of Zipf alpha on different algorithms
- Figure C: Throughput comparison
"""

import os
import csv
import numpy as np
import subprocess
import time
from pathlib import Path
from collections import defaultdict

# Try to use non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuration
RESULT_DIR = Path("/home/ftj/NSDI24-SIEVE/experiment_results")
FIGURE_DIR = Path("/home/ftj/NSDI24-SIEVE/experiment_results/figures")
CACHESIM_PATH = Path("/home/ftj/NSDI24-SIEVE/libCacheSim/_build/bin/cachesim")
TRACE_DIR = Path("/home/ftj/NSDI24-SIEVE/mydata/zipf")

# Style settings for academic papers
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Algorithm display names and colors
ALGO_STYLES = {
    'fifo': {'label': 'FIFO', 'color': '#1f77b4', 'marker': 'o'},
    'lru': {'label': 'LRU', 'color': '#ff7f0e', 'marker': 's'},
    'clock': {'label': 'Clock', 'color': '#2ca02c', 'marker': '^'},
    'sieve': {'label': 'SIEVE', 'color': '#d62728', 'marker': 'D'},
    'ghostsieve': {'label': 'GhostSIEVE', 'color': '#9467bd', 'marker': 'v'},
}


def load_results(csv_path):
    """Load experiment results from CSV file"""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'trace_name': row['trace_name'],
                'algorithm': row['algorithm'],
                'cache_size': int(row['cache_size']),
                'miss_ratio': float(row['miss_ratio']),
                'hit_ratio': float(row['hit_ratio'])
            })
    return results


def plot_hit_rate_vs_cache_size(results, output_path):
    """Figure A: Hit Rate vs Cache Size for each Zipf alpha"""
    zipf_traces = sorted(set(r['trace_name'] for r in results if r['trace_name'].startswith('zipf')))
    
    fig, axes = plt.subplots(1, len(zipf_traces), figsize=(5*len(zipf_traces), 5), sharey=True)
    if len(zipf_traces) == 1:
        axes = [axes]
    
    for ax, trace in zip(axes, zipf_traces):
        alpha = trace.split('_')[1]
        trace_results = [r for r in results if r['trace_name'] == trace]
        
        for algo in ALGO_STYLES.keys():
            algo_results = [r for r in trace_results if r['algorithm'] == algo]
            if not algo_results:
                continue
            
            algo_results.sort(key=lambda x: x['cache_size'])
            sizes = [r['cache_size'] for r in algo_results]
            hits = [r['hit_ratio'] * 100 for r in algo_results]
            
            style = ALGO_STYLES[algo]
            ax.plot(sizes, hits, marker=style['marker'], color=style['color'],
                   label=style['label'], linewidth=2, markersize=8)
        
        ax.set_xlabel('Cache Size')
        ax.set_ylabel('Hit Rate (%)')
        ax.set_title(f'Zipf α = {alpha}')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 100)
    
    plt.suptitle('Hit Rate vs Cache Size under Different Zipf Distributions', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Saved: {output_path}")


def plot_zipf_alpha_comparison(results, output_path):
    """Figure B: Bar chart comparing algorithm performance at different Zipf alpha"""
    zipf_traces = sorted(set(r['trace_name'] for r in results if r['trace_name'].startswith('zipf')))
    algorithms = list(ALGO_STYLES.keys())
    
    # Use middle cache size for comparison
    target_size = 250  # Middle cache size
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(zipf_traces))
    width = 0.15
    multiplier = 0
    
    for algo in algorithms:
        hit_rates = []
        for trace in zipf_traces:
            trace_results = [r for r in results 
                           if r['trace_name'] == trace and r['algorithm'] == algo]
            # Find the closest cache size
            if trace_results:
                closest = min(trace_results, key=lambda x: abs(x['cache_size'] - target_size))
                hit_rates.append(closest['hit_ratio'] * 100)
            else:
                hit_rates.append(0)
        
        style = ALGO_STYLES[algo]
        offset = width * multiplier
        rects = ax.bar(x + offset, hit_rates, width, label=style['label'], color=style['color'])
        
        # Add value labels on bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        multiplier += 1
    
    ax.set_xlabel('Zipf Distribution Parameter (α)')
    ax.set_ylabel('Hit Rate (%)')
    ax.set_title(f'Algorithm Performance Comparison (Cache Size = {target_size})')
    ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax.set_xticklabels([t.replace('zipf_', 'α = ') for t in zipf_traces])
    ax.legend(loc='upper left', ncol=2)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Saved: {output_path}")


def measure_throughput(trace_path, algorithm, cache_size, num_runs=3):
    """Measure throughput for a given configuration"""
    cmd = [
        str(CACHESIM_PATH),
        str(trace_path),
        "txt",
        algorithm,
        str(cache_size),
        "--ignore-obj-size", "1",
        "--num-thread", "1"
    ]
    
    throughputs = []
    for _ in range(num_runs):
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            stdout = result.stdout.decode('utf-8')
            
            for line in stdout.split('\n'):
                if 'throughput' in line:
                    parts = line.split('throughput')
                    if len(parts) >= 2:
                        tp_str = parts[1].strip().split()[0]
                        throughput = float(tp_str)
                        throughputs.append(throughput)
                        break
        except Exception as e:
            print(f"[WARNING] Throughput measurement failed: {e}")
    
    if throughputs:
        return np.mean(throughputs), np.std(throughputs)
    return 0, 0


def plot_throughput_comparison(output_path):
    """Figure C: Throughput comparison across algorithms"""
    trace_path = TRACE_DIR / "zipf_1.0_small"
    if not trace_path.exists():
        print(f"[WARNING] Trace not found for throughput test: {trace_path}")
        return
    
    algorithms = list(ALGO_STYLES.keys())
    cache_size = 500
    
    print("[INFO] Measuring throughput (this may take a minute)...")
    throughputs = {}
    errors = {}
    
    for algo in algorithms:
        tp, err = measure_throughput(trace_path, algo, cache_size, num_runs=3)
        throughputs[algo] = tp
        errors[algo] = err
        print(f"  {algo}: {tp:.2f} ± {err:.2f} MQPS")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(algorithms))
    colors = [ALGO_STYLES[algo]['color'] for algo in algorithms]
    labels = [ALGO_STYLES[algo]['label'] for algo in algorithms]
    values = [throughputs[algo] for algo in algorithms]
    errs = [errors[algo] for algo in algorithms]
    
    bars = ax.bar(x, values, yerr=errs, capsize=5, color=colors, edgecolor='black')
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Throughput (MQPS)')
    ax.set_title(f'Throughput Comparison (Zipf α=1.0, Cache Size={cache_size})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Saved: {output_path}")


def plot_sieve_improvement(results, output_path):
    """Plot showing SIEVE vs GhostSIEVE improvement"""
    zipf_traces = sorted(set(r['trace_name'] for r in results if r['trace_name'].startswith('zipf')))
    
    fig, axes = plt.subplots(1, len(zipf_traces), figsize=(5*len(zipf_traces), 5), sharey=True)
    if len(zipf_traces) == 1:
        axes = [axes]
    
    for ax, trace in zip(axes, zipf_traces):
        alpha = trace.split('_')[1]
        trace_results = [r for r in results if r['trace_name'] == trace]
        
        # Only plot SIEVE and GhostSIEVE
        for algo in ['sieve', 'ghostsieve']:
            algo_results = [r for r in trace_results if r['algorithm'] == algo]
            if not algo_results:
                continue
            
            algo_results.sort(key=lambda x: x['cache_size'])
            sizes = [r['cache_size'] for r in algo_results]
            hits = [r['hit_ratio'] * 100 for r in algo_results]
            
            style = ALGO_STYLES[algo]
            ax.plot(sizes, hits, marker=style['marker'], color=style['color'],
                   label=style['label'], linewidth=2, markersize=8)
        
        ax.set_xlabel('Cache Size')
        ax.set_ylabel('Hit Rate (%)')
        ax.set_title(f'Zipf α = {alpha}')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 100)
        
        # Add improvement annotation
        sieve_results = [r for r in trace_results if r['algorithm'] == 'sieve']
        ghost_results = [r for r in trace_results if r['algorithm'] == 'ghostsieve']
        if sieve_results and ghost_results:
            sieve_avg = np.mean([r['hit_ratio'] for r in sieve_results])
            ghost_avg = np.mean([r['hit_ratio'] for r in ghost_results])
            diff = (sieve_avg - ghost_avg) * 100
            ax.text(0.05, 0.95, f'SIEVE advantage: {diff:.1f}%',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('SIEVE vs GhostSIEVE Performance Comparison', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Saved: {output_path}")


def load_cluster45_results(csv_path):
    """Load cluster45 results from CSV file"""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'trace_name': row['trace_name'],
                'algorithm': row['algorithm'],
                'cache_size_mb': int(row['cache_size_mb']),
                'miss_ratio': float(row['miss_ratio']),
                'hit_ratio': float(row['hit_ratio']),
                'throughput': float(row['throughput_mqps'])
            })
    return results


def plot_real_trace_results(results, output_path):
    """Plot Hit Rate vs Cache Size for real trace (cluster45)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo in ALGO_STYLES.keys():
        algo_results = [r for r in results if r['algorithm'] == algo]
        if not algo_results:
            continue
        
        algo_results.sort(key=lambda x: x['cache_size_mb'])
        sizes = [r['cache_size_mb'] for r in algo_results]
        hits = [r['hit_ratio'] * 100 for r in algo_results]
        
        style = ALGO_STYLES[algo]
        ax.plot(sizes, hits, marker=style['marker'], color=style['color'],
               label=style['label'], linewidth=2, markersize=8)
    
    ax.set_xlabel('Cache Size (MB)')
    ax.set_ylabel('Hit Rate (%)')
    ax.set_title('Real Trace (Twitter cluster45): Hit Rate vs Cache Size')
    ax.legend(loc='lower right')
    ax.set_ylim(45, 60)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Saved: {output_path}")


def plot_real_trace_throughput(results, output_path):
    """Plot throughput for real trace"""
    algorithms = list(ALGO_STYLES.keys())
    
    # Use 100MB cache size for comparison
    target_size = 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    throughputs = []
    for algo in algorithms:
        algo_results = [r for r in results if r['algorithm'] == algo]
        if algo_results:
            closest = min(algo_results, key=lambda x: abs(x['cache_size_mb'] - target_size))
            throughputs.append(closest['throughput'])
        else:
            throughputs.append(0)
    
    x = np.arange(len(algorithms))
    colors = [ALGO_STYLES[algo]['color'] for algo in algorithms]
    labels = [ALGO_STYLES[algo]['label'] for algo in algorithms]
    
    bars = ax.bar(x, throughputs, color=colors, edgecolor='black')
    
    for bar, val in zip(bars, throughputs):
        height = bar.get_height()
        ax.annotate(f'{val:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Throughput (MQPS)')
    ax.set_title(f'Real Trace Throughput (cluster45, Cache Size=100MB)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Saved: {output_path}")


def main():
    print("="*60)
    print("SIEVE Experiment Visualization")
    print("="*60)
    
    # Create figure directory
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load Zipf results
    csv_path = RESULT_DIR / "zipf_results.csv"
    if not csv_path.exists():
        print(f"[ERROR] Results not found: {csv_path}")
        print("Please run run_experiments.py first.")
        return
    
    results = load_results(csv_path)
    print(f"[INFO] Loaded {len(results)} Zipf results")
    
    # Generate Zipf figures
    print("\n[Phase 1] Generating Figure A: Hit Rate vs Cache Size...")
    plot_hit_rate_vs_cache_size(results, FIGURE_DIR / "figure_a_hit_rate_vs_cache_size.png")
    
    print("\n[Phase 2] Generating Figure B: Zipf Alpha Comparison...")
    plot_zipf_alpha_comparison(results, FIGURE_DIR / "figure_b_zipf_alpha_comparison.png")
    
    print("\n[Phase 3] Generating Figure C: Throughput Comparison...")
    plot_throughput_comparison(FIGURE_DIR / "figure_c_throughput_comparison.png")
    
    print("\n[Phase 4] Generating SIEVE Improvement Figure...")
    plot_sieve_improvement(results, FIGURE_DIR / "figure_d_sieve_improvement.png")
    
    # Load and plot real trace results
    cluster45_path = RESULT_DIR / "cluster45_results.csv"
    if cluster45_path.exists():
        print("\n[Phase 5] Generating Real Trace Figures...")
        cluster45_results = load_cluster45_results(cluster45_path)
        print(f"[INFO] Loaded {len(cluster45_results)} real trace results")
        
        plot_real_trace_results(cluster45_results, FIGURE_DIR / "figure_e_real_trace_hit_rate.png")
        plot_real_trace_throughput(cluster45_results, FIGURE_DIR / "figure_f_real_trace_throughput.png")
    else:
        print("\n[INFO] No real trace results found, skipping...")
    
    print("\n" + "="*60)
    print(f"All figures saved to: {FIGURE_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
