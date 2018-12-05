#! /bin/bash

for seed in $(seq 1 5); do
    eval "python -O main.py -n 3 -s $seed -bs 4 --norm_type instance -g 0 -e 15 --dataset edges2shoes --num_workers 4 --use_leaky2dc"
done
