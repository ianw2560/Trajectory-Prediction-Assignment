#!/bin/bash

#==============================================================================
# Test four different architechture configurations
#==============================================================================
python train_eval.py --embed_size 256 --enc_layers 2 --heads 8 --forward_expansion 2 -o baseline
python train_eval.py --embed_size 128 --enc_layers 2 --heads 4 --forward_expansion 2 -o small
python train_eval.py --embed_size 256 --enc_layers 4 --heads 8 --forward_expansion 2 -o deep
python train_eval.py --embed_size 512 --enc_layers 2 --heads 8 --forward_expansion 4 -o wide

#==============================================================================
# Use fully non-linear QVK projections
#==============================================================================
python train_eval.py --embed_size 256 --enc_layers 2 --heads 8 --forward_expansion 2 --v_is_twolayer true -o baseline_fnl
python train_eval.py --embed_size 128 --enc_layers 2 --heads 4 --forward_expansion 2 --v_is_twolayer true -o small_fnl
python train_eval.py --embed_size 256 --enc_layers 4 --heads 8 --forward_expansion 2 --v_is_twolayer true -o deep_fnl
python train_eval.py --embed_size 512 --enc_layers 2 --heads 8 --forward_expansion 4 --v_is_twolayer true -o wide_fnl

#==============================================================================
# Use dynamic K-means clustering
#==============================================================================
python train_eval.py --embed_size 256 --enc_layers 2 --heads 8 --forward_expansion 2 --use_dynamic_clustering -o baseline_dc
python train_eval.py --embed_size 128 --enc_layers 2 --heads 4 --forward_expansion 2 --use_dynamic_clustering -o small_dc
python train_eval.py --embed_size 256 --enc_layers 4 --heads 8 --forward_expansion 2 --use_dynamic_clustering -o deep_dc
python train_eval.py --embed_size 512 --enc_layers 2 --heads 8 --forward_expansion 4 --use_dynamic_clustering -o wide_dc

#==============================================================================
# Use both dynamic K-means clustering and fully non-linear QVK projections
#==============================================================================
python train_eval.py --embed_size 256 --enc_layers 2 --heads 8 --forward_expansion 2 --v_is_twolayer true --use_dynamic_clustering -o baseline_fnl_dc
python train_eval.py --embed_size 128 --enc_layers 2 --heads 4 --forward_expansion 2 --v_is_twolayer true --use_dynamic_clustering -o small_fnl_dc
python train_eval.py --embed_size 256 --enc_layers 4 --heads 8 --forward_expansion 2 --v_is_twolayer true --use_dynamic_clustering -o deep_fnl_dc
python train_eval.py --embed_size 512 --enc_layers 2 --heads 8 --forward_expansion 4 --v_is_twolayer true --use_dynamic_clustering -o wide_fnl_dc
