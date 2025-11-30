#!/bin/bash


# Test three different architechture configurations
python train_eval.py --embed_size 128 --enc_layers 2 --heads 4 --forward_expansion 2
python train_eval.py --embed_size 256 --enc_layers 4 --heads 8 --forward_expansion 2
python train_eval.py --embed_size 512 --enc_layers 2 --heads 8 --forward_expansion 4