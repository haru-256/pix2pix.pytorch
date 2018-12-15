#! /bin/bash

for seed in $(seq 1 5); do
    eval "python -O main.py -n 6 -s $seed -bs 4 --norm_type instance -g 0 -e 15 --dataset edges2shoes --num_workers 4 --not_affine"
done

