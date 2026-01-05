#!/usr/bin/env python3
"""
SIEVE vs LRU Cache Visualization - Step-by-Step Animation
==========================================================
This script creates a visual comparison showing the internal state
of SIEVE and LRU caches at each step of a carefully designed access pattern.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import OrderedDict
from pathlib import Path

FIGURE_DIR = Path("/home/ftj/NSDI24-SIEVE/experiment_results/figures")

# ============================================================================
# Cache Implementations
# ============================================================================

class VisualSIEVE:
    """SIEVE with full state tracking for visualization"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()  # obj_id -> visited (True/False)
        self.hand_pos = None  # Position of the "hand" (index from head)
        self.hits = 0
        self.misses = 0
        self.states = []  # Record state after each access
    
    def access(self, obj_id):
        evicted = None
        action = ""
        
        if obj_id in self.cache:
            # HIT: just set visited bit, don't move
            self.cache[obj_id] = True
            self.hits += 1
            action = "HIT"
        else:
            # MISS
            self.misses += 1
            
            if len(self.cache) >= self.capacity:
                evicted = self._evict()
                action = f"MISS→evict({evicted})"
            else:
                action = "MISS→insert"
            
            # Insert at head with visited=False
            new_cache = OrderedDict()
            new_cache[obj_id] = False
            new_cache.update(self.cache)
            self.cache = new_cache
        
        # Record state
        self.states.append({
            'access': obj_id,
            'action': action,
            'cache': [(k, v) for k, v in self.cache.items()],
            'hand': self.hand_pos,
            'evicted': evicted
        })
        
        return action
    
    def _evict(self):
        """SIEVE eviction with hand tracking"""
        keys = list(self.cache.keys())
        n = len(keys)
        
        # Start from hand position or tail
        if self.hand_pos is None or self.hand_pos >= n:
            idx = n - 1
        else:
            idx = self.hand_pos
        
        start_idx = idx
        while True:
            key = keys[idx]
            if self.cache[key]:
                # Visited: clear and move hand
                self.cache[key] = False
                idx = idx - 1 if idx > 0 else n - 1
            else:
                # Not visited: evict this one
                self.hand_pos = idx - 1 if idx > 0 else None
                del self.cache[key]
                return key
            
            if idx == start_idx:
                # Full circle - evict current
                self.hand_pos = idx - 1 if idx > 0 else None
                del self.cache[keys[idx]]
                return keys[idx]


class VisualLRU:
    """LRU with full state tracking for visualization"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.states = []
    
    def access(self, obj_id):
        evicted = None
        action = ""
        
        if obj_id in self.cache:
            # HIT: move to front (MRU position)
            self.cache.move_to_end(obj_id, last=False)
            self.hits += 1
            action = "HIT→move to front"
        else:
            # MISS
            self.misses += 1
            
            if len(self.cache) >= self.capacity:
                # Evict LRU (last item)
                evicted = next(reversed(self.cache))
                del self.cache[evicted]
                action = f"MISS→evict({evicted})"
            else:
                action = "MISS→insert"
            
            # Insert at front
            new_cache = OrderedDict()
            new_cache[obj_id] = True
            new_cache.update(self.cache)
            self.cache = new_cache
        
        self.states.append({
            'access': obj_id,
            'action': action,
            'cache': list(self.cache.keys()),
            'evicted': evicted
        })
        
        return action


def draw_cache_state(ax, cache_data, title, is_sieve=True, hand_pos=None, 
                     highlight_access=None, evicted=None, step_num=0, action=""):
    """Draw a single cache state as boxes"""
    ax.clear()
    ax.set_xlim(-1, 5.5)
    ax.set_ylim(-1, 2.5)
    ax.axis('off')
    
    # Calculate center offset to align with title
    n_boxes = len(cache_data) if cache_data else 3
    box_width = 0.9
    total_width = n_boxes * box_width + (n_boxes - 1) * 0.1
    start_x = (4.5 - total_width) / 2  # Center in the available space
    
    # Title - centered
    center_x = 2.25
    ax.text(center_x, 2.2, title, fontsize=14, fontweight='bold', ha='center')
    ax.text(center_x, 1.8, f"Step {step_num}: {action}", fontsize=10, ha='center', 
            color='darkblue')
    
    # Draw cache slots
    box_height = 0.8
    
    for i, item in enumerate(cache_data):
        if is_sieve:
            obj_id, visited = item
            label = f"{obj_id}"
            sublabel = "V" if visited else "-"
        else:
            obj_id = item
            label = f"{obj_id}"
            sublabel = ""
        
        x = start_x + i * (box_width + 0.1)
        y = 0.5
        
        # Determine box color
        if obj_id == highlight_access:
            if "HIT" in action:
                color = '#90EE90'  # Light green for hit
            else:
                color = '#FFB6C1'  # Light red for miss (new insert)
        elif obj_id == evicted:
            color = '#FF6B6B'  # Red for evicted
        else:
            color = '#E8E8E8'  # Gray for others
        
        # Draw box
        rect = mpatches.FancyBboxPatch((x, y), box_width, box_height,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor='black',
                                        linewidth=2)
        ax.add_patch(rect)
        
        # Object ID
        ax.text(x + box_width/2, y + box_height/2 + 0.1, label, 
               fontsize=14, fontweight='bold', ha='center', va='center')
        
        # Visited bit for SIEVE
        if is_sieve and sublabel:
            ax.text(x + box_width/2, y + 0.15, f"({sublabel})", 
                   fontsize=10, ha='center', va='center', color='gray')
        
        # Hand pointer for SIEVE
        if is_sieve and hand_pos is not None and i == hand_pos:
            ax.annotate('', xy=(x + box_width/2, y), 
                       xytext=(x + box_width/2, y - 0.4),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
            ax.text(x + box_width/2, y - 0.6, 'hand', fontsize=9, 
                   ha='center', color='red')
    
    # Labels - positioned relative to boxes
    if cache_data:
        first_x = start_x
        last_x = start_x + (len(cache_data) - 1) * (box_width + 0.1) + box_width
        ax.text(first_x - 0.15, 0.9, 'HEAD\n(MRU)', fontsize=8, ha='right', va='center')
        ax.text(last_x + 0.15, 0.9, 'TAIL\n(LRU)', fontsize=8, ha='left', va='center')
    
    # Legend for SIEVE
    if is_sieve:
        ax.text(start_x, -0.5, "V=visited, -=not visited", fontsize=8, color='gray')


def create_step_by_step_visualization():
    """Create the main visualization showing SIEVE vs LRU step by step"""
    
    print("="*60)
    print("Creating Step-by-Step Cache Visualization")
    print("="*60)
    
    # Design access pattern to clearly show SIEVE advantage
    # 1. Establish working set A, B, C with repeated access (builds up visited bits)
    # 2. Scan with X, Y (one-time access - should be evicted quickly in SIEVE)
    # 3. Return to working set (SIEVE should hit, LRU should miss)
    
    cache_size = 3
    accesses = ['A', 'B', 'C', 'A', 'B', 'X', 'Y', 'A', 'B', 'C']
    
    print(f"\nCache size: {cache_size}")
    print(f"Access pattern: {accesses}")
    print("Working set: A, B, C")
    print("Scan objects: X, Y")
    
    # Run both algorithms
    sieve = VisualSIEVE(cache_size)
    lru = VisualLRU(cache_size)
    
    for obj in accesses:
        sieve.access(obj)
        lru.access(obj)
    
    # Create figure with all steps
    n_steps = len(accesses)
    fig, axes = plt.subplots(n_steps, 2, figsize=(12, 3 * n_steps))
    
    for i in range(n_steps):
        s_state = sieve.states[i]
        l_state = lru.states[i]
        
        # SIEVE (left column)
        draw_cache_state(
            axes[i, 0], 
            s_state['cache'],
            "SIEVE",
            is_sieve=True,
            hand_pos=s_state['hand'],
            highlight_access=s_state['access'],
            evicted=s_state['evicted'],
            step_num=i+1,
            action=f"Access '{s_state['access']}' → {s_state['action']}"
        )
        
        # LRU (right column)
        draw_cache_state(
            axes[i, 1],
            l_state['cache'],
            "LRU", 
            is_sieve=False,
            highlight_access=l_state['access'],
            evicted=l_state['evicted'],
            step_num=i+1,
            action=f"Access '{l_state['access']}' → {l_state['action']}"
        )
    
    plt.suptitle("SIEVE vs LRU: Step-by-Step Cache State Comparison", 
                fontsize=16, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    output_path = FIGURE_DIR / "case_study_step_by_step.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[INFO] Saved: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"SIEVE: {sieve.hits} hits, {sieve.misses} misses, "
          f"hit rate = {sieve.hits/len(accesses)*100:.1f}%")
    print(f"LRU:   {lru.hits} hits, {lru.misses} misses, "
          f"hit rate = {lru.hits/len(accesses)*100:.1f}%")
    
    # Create compact summary figure
    create_summary_figure(sieve, lru, accesses)


def create_summary_figure(sieve, lru, accesses):
    """Create a compact summary showing key differences"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Hit/Miss timeline
    ax1 = axes[0]
    x = range(1, len(accesses) + 1)
    
    sieve_cumulative = []
    lru_cumulative = []
    s_hits = l_hits = 0
    
    for i, (s, l) in enumerate(zip(sieve.states, lru.states)):
        if "HIT" in s['action']:
            s_hits += 1
        if "HIT" in l['action']:
            l_hits += 1
        sieve_cumulative.append(s_hits / (i + 1) * 100)
        lru_cumulative.append(l_hits / (i + 1) * 100)
    
    ax1.plot(x, sieve_cumulative, 'ro-', label='SIEVE', linewidth=2, markersize=10)
    ax1.plot(x, lru_cumulative, 'bs-', label='LRU', linewidth=2, markersize=10)
    
    # Mark phases
    ax1.axvspan(0.5, 6.5, alpha=0.2, color='green', label='Working Set')
    ax1.axvspan(6.5, 8.5, alpha=0.2, color='red', label='Scan')
    ax1.axvspan(8.5, 11.5, alpha=0.2, color='green')
    
    # Add access labels
    for i, acc in enumerate(accesses):
        ax1.annotate(acc, (i + 1, 5), fontsize=9, ha='center')
    
    ax1.set_xlabel('Request Number', fontsize=12)
    ax1.set_ylabel('Cumulative Hit Rate (%)', fontsize=12)
    ax1.set_title('Hit Rate Over Time', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.grid(True, alpha=0.3)
    
    # Right: Key insight diagram
    ax2 = axes[1]
    ax2.axis('off')
    
    text = """
    Key Insight: Why SIEVE Outperforms LRU
    ══════════════════════════════════════
    
    LRU Problem:
    • On HIT, LRU moves object to front
    • Scan objects (X, Y) push working set to tail
    • Working set objects (A, B, C) get evicted!
    
    SIEVE Solution:
    • On HIT, SIEVE only sets visited bit (no movement)
    • Working set stays protected with visited=1
    • Scan objects (visited=0) are quickly evicted
    • When returning to working set → HIT!
    
    Result:
    • SIEVE: {} hits / {} requests = {:.0f}% hit rate
    • LRU:   {} hits / {} requests = {:.0f}% hit rate
    • SIEVE advantage: +{:.0f}%
    """.format(
        sieve.hits, len(accesses), sieve.hits/len(accesses)*100,
        lru.hits, len(accesses), lru.hits/len(accesses)*100,
        (sieve.hits - lru.hits)/len(accesses)*100
    )
    
    ax2.text(0.1, 0.95, text, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle("SIEVE vs LRU: Performance Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = FIGURE_DIR / "case_study_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {output_path}")


if __name__ == "__main__":
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    create_step_by_step_visualization()
