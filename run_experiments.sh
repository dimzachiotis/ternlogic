#!/bin/bash

# Configuration
seeds=(0 1 2)
tau_val=20 
k_values=(64000 32000 16000)
l_val=(8 6 4 2) 
grad_fact=(1.0 2.0)
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# Corrected loop: iterate through tau_values array
for g_f in "${grad_fact[@]}"; do
    for k in "${k_values[@]}"; do
        for l in "${l_val[@]}"; do
        
            # Calculate kl product
            kl=$((k * l))

            # Dynamic max_jobs logic
            if [ "$kl" -ge $((64000 * 6)) ]; then
                max_jobs=2
            elif [ "$kl" -ge $((32000 * 6)) ]; then
                max_jobs=4
            elif [ "$kl" -ge $((16000 * 6)) ]; then
                max_jobs=6
            else
                max_jobs=8
            fi

            # Define log file (using 't' loop variable)
            log_file="$LOG_DIR/gradfact${g_f}_k${k}_l${l}.log"

            cmd="python experiments/main.py \
                -bs 100 -t $tau_val --dataset mnist -ni 200000 -ef 1000 \
                -k $k -l $l --compile_model --implementation cuda \
                -lr 0.01 --grad-factor $g_f"

            # Execute in background
            echo "Starting Experiment: gradfactor=$g_f, k=$k,l=$l  (Max Parallel: $max_jobs)"
            bash -c "$cmd" > "$log_file" 2>&1 &

            # Concurrency Management
            while [ $(jobs -rp | wc -l) -ge "$max_jobs" ]; do
                sleep 2
            done
        done
    done
done

wait
echo "All experiments completed."