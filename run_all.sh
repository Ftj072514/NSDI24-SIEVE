#!/bin/bash
# =============================================================================
# SIEVE Cache Algorithm - One-Click Experiment Runner
# =============================================================================
# This script reproduces all experiments from the SIEVE cache algorithm study.
# 
# Usage: ./run_all.sh [OPTIONS]
#   --quick    : Run quick experiments only (5 min)
#   --full     : Run full experiments including real trace (30 min)
#   --plot     : Generate plots only (requires existing results)
#   --clean    : Clean all generated data and start fresh
#
# Output:
#   - experiment_results/zipf_full_results.csv
#   - experiment_results/cluster45_results.csv
#   - experiment_results/figures/*.png
#   - experiment_results/REPORT.md
# =============================================================================

set -e
cd "$(dirname "$0")"

CACHESIM="./libCacheSim/_build/bin/cachesim"
DATA_GEN="./libCacheSim/scripts/data_gen.py"
TRACE_DIR="./mydata/zipf"
RESULT_DIR="./experiment_results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_header() {
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${GREEN}============================================${NC}"
}

print_step() {
    echo -e "${YELLOW}[STEP]${NC} $1"
}

# Parse arguments
MODE="quick"
for arg in "$@"; do
    case $arg in
        --quick) MODE="quick" ;;
        --full) MODE="full" ;;
        --plot) MODE="plot" ;;
        --clean) 
            echo "Cleaning generated data..."
            rm -rf "$TRACE_DIR"/* "$RESULT_DIR"/*
            echo "Done."
            exit 0
            ;;
        --help|-h)
            echo "Usage: $0 [--quick|--full|--plot|--clean]"
            exit 0
            ;;
    esac
done

# Check prerequisites
if [ ! -f "$CACHESIM" ]; then
    echo -e "${RED}Error: cachesim not found at $CACHESIM${NC}"
    echo "Please build libCacheSim first:"
    echo "  cd libCacheSim && mkdir -p _build && cd _build && cmake .. && make -j4"
    exit 1
fi

mkdir -p "$TRACE_DIR" "$RESULT_DIR/figures"

# =============================================================================
# Phase 1: Generate Zipf traces
# =============================================================================
if [ "$MODE" != "plot" ]; then
    print_header "Phase 1: Generating Zipf Traces"
    
    if [ "$MODE" = "quick" ]; then
        ALPHAS="0.8 1.0 1.2"
        NUM_REQ=500000
        NUM_OBJ=5000
    else
        ALPHAS="0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4"
        NUM_REQ=1000000
        NUM_OBJ=10000
    fi
    
    for alpha in $ALPHAS; do
        trace_file="$TRACE_DIR/zipf_${alpha}"
        if [ ! -f "$trace_file" ]; then
            print_step "Generating Zipf trace (α=$alpha)..."
            python3 "$DATA_GEN" -m $NUM_OBJ -n $NUM_REQ --alpha $alpha > "$trace_file"
        else
            echo "  Trace exists: $trace_file"
        fi
    done
fi

# =============================================================================
# Phase 2: Run Zipf experiments
# =============================================================================
if [ "$MODE" != "plot" ]; then
    print_header "Phase 2: Running Zipf Experiments"
    
    RESULT_FILE="$RESULT_DIR/zipf_full_results.csv"
    echo "alpha,algorithm,cache_size,miss_ratio,hit_ratio" > "$RESULT_FILE"
    
    ALGOS="fifo lru clock sieve ghostsieve"
    
    if [ "$MODE" = "quick" ]; then
        SIZES="100 500 1000"
    else
        SIZES="50 100 250 500 1000"
    fi
    
    for alpha in $ALPHAS; do
        print_step "Testing α=$alpha..."
        trace_file="$TRACE_DIR/zipf_${alpha}"
        
        for algo in $ALGOS; do
            for size in $SIZES; do
                result=$("$CACHESIM" "$trace_file" txt $algo $size --ignore-obj-size 1 --num-thread 1 2>&1 | grep "miss ratio")
                miss=$(echo "$result" | sed 's/.*miss ratio \([0-9.]*\).*/\1/')
                hit=$(echo "scale=4; 1 - $miss" | bc)
                echo "$alpha,$algo,$size,$miss,$hit" >> "$RESULT_FILE"
            done
        done
    done
    
    echo -e "${GREEN}Saved: $RESULT_FILE${NC}"
fi

# =============================================================================
# Phase 3: Run real trace experiments (full mode only)
# =============================================================================
if [ "$MODE" = "full" ]; then
    print_header "Phase 3: Running Real Trace Experiments"
    
    REAL_TRACE="./cluster45.oracleGeneral.zst"
    if [ -f "$REAL_TRACE" ]; then
        RESULT_FILE="$RESULT_DIR/cluster45_results.csv"
        echo "trace_name,algorithm,cache_size,cache_size_mb,miss_ratio,hit_ratio,throughput_mqps" > "$RESULT_FILE"
        
        for algo in fifo lru clock sieve ghostsieve; do
            for size_mb in 10 50 100 200 500; do
                print_step "Testing $algo at ${size_mb}MB..."
                result=$("$CACHESIM" "$REAL_TRACE" oracleGeneral $algo ${size_mb}MB --num-thread 1 -n 5000000 2>&1 | grep "miss ratio")
                miss=$(echo "$result" | sed 's/.*miss ratio \([0-9.]*\).*/\1/')
                tp=$(echo "$result" | sed 's/.*throughput \([0-9.]*\).*/\1/')
                hit=$(echo "scale=4; 1 - $miss" | bc)
                cache_bytes=$((size_mb * 1048576))
                echo "cluster45,$algo,$cache_bytes,$size_mb,$miss,$hit,$tp" >> "$RESULT_FILE"
            done
        done
        
        echo -e "${GREEN}Saved: $RESULT_FILE${NC}"
    else
        echo -e "${YELLOW}Warning: Real trace not found, skipping...${NC}"
    fi
fi

# =============================================================================
# Phase 4: Generate plots
# =============================================================================
print_header "Phase 4: Generating Visualizations"
python3 plot_results.py

# =============================================================================
# Phase 5: Run Case Study
# =============================================================================
print_header "Phase 5: Running Case Study"
python3 case_study.py

print_header "All Done!"
echo ""
echo "Results saved to:"
echo "  - $RESULT_DIR/zipf_full_results.csv"
echo "  - $RESULT_DIR/figures/*.png"
echo "  - $RESULT_DIR/REPORT.md"
echo ""
echo "To view the report:"
echo "  cat $RESULT_DIR/REPORT.md"
