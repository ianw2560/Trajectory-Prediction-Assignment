#!/bin/bash

#==============================================================================
# Test three different architechture configurations
#==============================================================================

# Baseline 
python train_eval.py --embed_size 256 --enc_layers 2 --heads 8 --forward_expansion 2 -o arch_baseline

# Small
python train_eval.py --embed_size 128 --enc_layers 2 --heads 4 --forward_expansion 2 -o arch_small

# Deep
python train_eval.py --embed_size 256 --enc_layers 4 --heads 8 --forward_expansion 2 -o arch_deep

# Wide
python train_eval.py --embed_size 512 --enc_layers 2 --heads 8 --forward_expansion 4 -o arch_wide

#==============================================================================
# Make V use two layers
#==============================================================================
python train_eval.py --v_is_twolayer true -o v_is_two_layer

#==============================================================================
# Make V use two layers
#==============================================================================
python train_eval.py --embed_size 256 --enc_layers 2 --heads 8 --forward_expansion 2 --use_dynamic_clustering -o use_dynamic_clustering_baseline
python train_eval.py --embed_size 128 --enc_layers 2 --heads 4 --forward_expansion 2 --use_dynamic_clustering -o use_dynamic_clustering_small
python train_eval.py --embed_size 256 --enc_layers 4 --heads 8 --forward_expansion 2 --use_dynamic_clustering -o use_dynamic_clustering_deep
python train_eval.py --embed_size 512 --enc_layers 2 --heads 8 --forward_expansion 4 --use_dynamic_clustering -o use_dynamic_clustering_wide
