#! /bin/bash

for seed in $(seq 4 10); do
    eval "python -0 main.py -n 1 -s $seed -bs 4 --norm_type batch -g 0 -e 15 --dataset edges2shoes --num_workers 4"
done
