#!/usr/bin/env python3
"""
SIEVE Cache Algorithm Experiment - Memory Efficient Version
============================================================
Optimized for low memory usage:
- Sequential algorithm execution (one at a time)
- Smaller trace files
- Single-threaded operation
"""

import os
import sys
import subprocess
import time
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Configuration
CACHESIM_PATH = Path("/home/ftj/NSDI24-SIEVE/libCacheSim/_build/bin/cachesim")
DATA_GEN_PATH = Path("/home/ftj/NSDI24-SIEVE/libCacheSim/scripts/data_gen.py")
TRACE_DIR = Path("/home/ftj/NSDI24-SIEVE/mydata")
RESULT_DIR = Path("/home/ftj/NSDI24-SIEVE/experiment_results")
ZIPF_DIR = TRACE_DIR / "zipf"

# Memory-efficient parameters
ALGORITHMS = ["fifo", "lru", "clock", "sieve", "ghostsieve"]
ZIPF_ALPHAS = [0.8, 1.0, 1.2]
# Smaller cache sizes for efficiency
CACHE_SIZE_RATIOS = [0.01, 0.02, 0.05, 0.10, 0.20]
# Much smaller trace size to avoid memory issues
ZIPF_NUM_OBJECTS = 5000     # Reduced from 10000
ZIPF_NUM_REQUESTS = 100000  # Reduced from 500000

@dataclass
class ExperimentResult:
    trace_name: str
    algorithm: str
    cache_size: int
    miss_ratio: float
    hit_ratio: float


def setup_directories():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ZIPF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Results will be saved to: {RESULT_DIR}")


def generate_zipf_trace(alpha: float) -> Path:
    """Generate a small Zipf distribution trace file"""
    trace_path = ZIPF_DIR / f"zipf_{alpha}_small"
    
    if trace_path.exists():
        print(f"[INFO] Trace exists: {trace_path}")
        return trace_path
    
    print(f"[INFO] Generating Zipf trace (alpha={alpha})...")
    
    cmd = [
        sys.executable, str(DATA_GEN_PATH),
        "-m", str(ZIPF_NUM_OBJECTS),
        "-n", str(ZIPF_NUM_REQUESTS),
        "--alpha", str(alpha)
    ]
    
    with open(trace_path, 'w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, timeout=120)
        if result.returncode != 0:
            print(f"[ERROR] Failed: {result.stderr.decode()}")
            return None
    
    print(f"[INFO] Generated: {trace_path}")
    return trace_path


def run_single_experiment(trace_path: str, algorithm: str, cache_size: int,
                         trace_format: str = "txt") -> Tuple[float, float]:
    """
    Run a single cachesim experiment (memory efficient)
    Returns: (miss_ratio, runtime_seconds)
    """
    cmd = [
        str(CACHESIM_PATH),
        trace_path,
        trace_format,
        algorithm,
        str(cache_size),
        "--ignore-obj-size", "1",
        "--num-thread", "1"  # Single thread for memory efficiency
    ]
    
    try:
        start = time.time()
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=180)
        elapsed = time.time() - start
        
        stdout = result.stdout.decode('utf-8')
        
        for line in stdout.split('\n'):
            # Parse format: "... miss ratio 0.4475, throughput ..."
            if 'miss ratio' in line:
                try:
                    idx = line.index('miss ratio')
                    parts = line[idx:].split()
                    if len(parts) >= 3:
                        miss_ratio = float(parts[2].strip(','))
                        return miss_ratio, elapsed
                except (ValueError, IndexError):
                    pass
        
        return None, elapsed
        
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Timeout")
        return None, 0
    except Exception as e:
        print(f"[ERROR] {e}")
        return None, 0


def run_zipf_experiments() -> List[ExperimentResult]:
    """Run experiments with Zipf traces sequentially"""
    results = []
    
    for alpha in ZIPF_ALPHAS:
        print(f"\n{'='*50}")
        print(f"Zipf alpha = {alpha}")
        print('='*50)
        
        trace_path = generate_zipf_trace(alpha)
        if trace_path is None:
            continue
        
        cache_sizes = [int(ZIPF_NUM_OBJECTS * r) for r in CACHE_SIZE_RATIOS]
        cache_sizes = [max(s, 50) for s in cache_sizes]
        
        trace_name = f"zipf_{alpha}"
        
        for algo in ALGORITHMS:
            print(f"  Algorithm: {algo}")
            for cs in cache_sizes:
                miss_ratio, elapsed = run_single_experiment(
                    str(trace_path), algo, cs
                )
                
                if miss_ratio is not None:
                    results.append(ExperimentResult(
                        trace_name=trace_name,
                        algorithm=algo,
                        cache_size=cs,
                        miss_ratio=miss_ratio,
                        hit_ratio=1.0 - miss_ratio
                    ))
                    print(f"    size={cs:5d}: miss={miss_ratio:.4f}, hit={1-miss_ratio:.4f}")
    
    return results


def run_real_trace_sample() -> List[ExperimentResult]:
    """Run experiments with a sample of real trace"""
    results = []
    
    # Use the existing sample file
    trace_path = "/home/ftj/NSDI24-SIEVE/cluster10.oracleGeneral.sample10"
    
    if not Path(trace_path).exists():
        print(f"[WARNING] Real trace not found")
        return results
    
    print(f"\n{'='*50}")
    print(f"Real trace: cluster10.sample10")
    print('='*50)
    
    # Very conservative cache sizes for real trace
    cache_sizes = [500, 1000, 2000, 5000, 10000]
    trace_name = "cluster10_sample"
    
    for algo in ALGORITHMS:
        print(f"  Algorithm: {algo}")
        for cs in cache_sizes:
            miss_ratio, elapsed = run_single_experiment(
                trace_path, algo, cs, trace_format="oracleGeneral"
            )
            
            if miss_ratio is not None:
                results.append(ExperimentResult(
                    trace_name=trace_name,
                    algorithm=algo,
                    cache_size=cs,
                    miss_ratio=miss_ratio,
                    hit_ratio=1.0 - miss_ratio
                ))
                print(f"    size={cs:5d}: miss={miss_ratio:.4f}")
    
    return results


def save_results_csv(results: List[ExperimentResult], filename: str):
    csv_path = RESULT_DIR / filename
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['trace_name', 'algorithm', 'cache_size', 'miss_ratio', 'hit_ratio'])
        for r in results:
            writer.writerow([r.trace_name, r.algorithm, r.cache_size, 
                           f"{r.miss_ratio:.6f}", f"{r.hit_ratio:.6f}"])
    
    print(f"[INFO] Saved: {csv_path}")
    return csv_path


def main():
    print("="*60)
    print("SIEVE Experiment Suite (Memory Efficient)")
    print("="*60)
    
    setup_directories()
    
    # Run Zipf experiments
    print("\n[Phase 1] Zipf experiments...")
    zipf_results = run_zipf_experiments()
    if zipf_results:
        save_results_csv(zipf_results, "zipf_results.csv")
    
    # Run real trace (sample only)
    print("\n[Phase 2] Real trace experiments...")
    real_results = run_real_trace_sample()
    if real_results:
        save_results_csv(real_results, "real_trace_results.csv")
    
    # Combined results
    all_results = zipf_results + real_results
    if all_results:
        save_results_csv(all_results, "all_results.csv")
    
    print("\n" + "="*60)
    print(f"Done! Total experiments: {len(all_results)}")
    print("="*60)
    
    return all_results


if __name__ == "__main__":
    results = main()
