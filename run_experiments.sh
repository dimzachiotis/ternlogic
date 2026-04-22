#!/bin/bash

dataset=mnist

for seed in 0 1 2; do
  for k in 64000 32000 16000 8000; do
    for l in 1 2 4 6; do
      kl=$((k * l))

      if   [ "$kl" -ge $((64000*6)) ]; then max_jobs=1
      elif [ "$kl" -ge $((32000*4)) ]; then max_jobs=2
      elif [ "$kl" -ge $((16000*4)) ]; then max_jobs=4
      else                                  max_jobs=8
      fi

      # For very heavy jobs wait for everything to finish first
      if [ "$max_jobs" -eq 1 ]; then
        wait
        echo "Cleared all jobs, starting heavy: k=$k l=$l seed=$seed"
      fi

      # Wait until a slot is free
      while [ "$(jobs -r | wc -l)" -ge "$max_jobs" ]; do
        sleep 5
      done

      echo "Starting: seed=$seed k=$k l=$l kl=$kl max_jobs=$max_jobs"

      python experiments/main.py \
        -bs 100 \
        -t 30 \
        --dataset $dataset \
        -ni 200000 \
        -ef 1000 \
        -k $k \
        -l $l \
        --compile_model \
        --implementation cuda \
        --seed $seed \
        --connections unique &

    done
  done
done

wait
echo "All experiments finished!"
