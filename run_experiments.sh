#!/bin/bash

# Configuration
seeds=(0 1 2)
k_values=(64000 32000 16000 8000 4000 2000)
l_values=(1 2 4 6)
lr_values=(0.01 0.007 0.005 0.003 0.001)
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

for lr in "${lr_values[@]}"; do
    for k in "${k_values[@]}"; do
        for l in "${l_values[@]}"; do

            
            # Calculate kl product
            kl=$((k * l))

            # Dynamic max_jobs logic
            if [ "$kl" -ge $((64000 * 6)) ]; then
                max_jobs=1
            elif [ "$kl" -ge $((32000 * 6)) ]; then
                max_jobs=2
            elif [ "$kl" -ge $((16000 * 6)) ]; then
                max_jobs=4
            else
                max_jobs=8
            fi

            # Define log file for this specific separate bash
            log_file="$LOG_DIR/exp_lr${lr}_k${k}_l${l}.log"

            # Command string
            cmd="python experiments/main.py \
                -bs 100 -t 30 --dataset mnist -ni 200000 -ef 1000 \
                -k $k -l $l --compile_model --implementation cuda \
                -lr $lr "

            # Execute in a separate bash background process
            # Output is redirected to the log file so you can inspect it later
            bash -c "echo 'Starting Experiment...'; $cmd" > "$log_file" 2>&1 &

            # Concurrency Management
            while [ $(jobs -rp | wc -l) -ge "$max_jobs" ]; do
                sleep 2
            done
        done
    done
done

wait
echo "All experiments completed."