#!/bin/bash

# Configuration
seeds=(1 2)
tau_val=30 
k_values=(64000 32000 16000 8000 4000 2000)
l_val=(6 4 2 1) 
lr_val=(0.01 0.007 0.005 0.003 0.001)
LOG_DIR="./logs"

mkdir -p "$LOG_DIR"

for sd in "${seeds[@]}"; do
    for k in "${k_values[@]}"; do
        for l in "${l_val[@]}"; do
            
            # --- Resource Management ---
            # Calculate resource pressure once per k/l group
            kl=$((k * l))
            
            if [ "$kl" -ge $((64000 * 6)) ]; then
                max_jobs=3
            elif [ "$kl" -ge $((32000 * 6)) ]; then
                max_jobs=5
            elif [ "$kl" -ge $((16000 * 6)) ]; then
                max_jobs=7
            else
                max_jobs=9
            fi

            for lr in "${lr_val[@]}"; do
                
                log_file="$LOG_DIR/seed${sd}_k${k}_l${l}_lr${lr}.log"

                # SKIP check: Don't waste time/GPU on completed experiments
                if [ -f "$log_file" ]; then
                    echo "Skipping existing: $log_file"
                    continue
                fi

                echo "Launching: seed=$sd, k=$k, l=$l, lr=$lr (Max Parallel: $max_jobs)"

                # Direct execution handles quotes and signals better than bash -c
                python experiments/main.py \
                    -bs 100 \
                    -t "$tau_val" \
                    --dataset mnist \
                    -ni 200000 \
                    -ef 1000 \
                    -k "$k" \
                    -l "$l" \
                    --compile_model \
                    --implementation cuda \
                    -lr "$lr" \
                    --seed "$sd" > "$log_file" 2>&1 &

                # --- Concurrency Control ---
                while [ $(jobs -rp | wc -l) -ge "$max_jobs" ]; do
                    sleep 2
                done
            done
        done
    done
done

wait
echo "Grid search completed successfully."