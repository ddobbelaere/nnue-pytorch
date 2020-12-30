#!/bin/bash

python train.py ../data2/noob14_d12_mpv4_rescore_d2.binpack ../data2/noob14_d12_mpv4_rescore_d2.binpack --gpus 1 --val_check_interval 2000 --threads 2 --batch-size 16384 --progress_bar_refresh_rate 20 --smart-fen-skipping --random-fen-skipping 7 --features=HalfKP^ --lambda=1.0 --max_epochs=100
