#!/bin/bash

#SBATCH --job-name=traj_analysis
#SBATCH --gres=gpu:nvidia_h100_pcie:1    
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --array=0-15
#SBATCH -o logs/traj_analysis_%A_%a.out

# List of commands
CMDS=(
  "python train_eval.py --embed_size 256 --enc_layers 2 --heads 8 --forward_expansion 2 -o baseline"
  "python train_eval.py --embed_size 128 --enc_layers 2 --heads 4 --forward_expansion 2 -o small"
  "python train_eval.py --embed_size 256 --enc_layers 4 --heads 8 --forward_expansion 2 -o deep"
  "python train_eval.py --embed_size 512 --enc_layers 2 --heads 8 --forward_expansion 4 -o wide"

  "python train_eval.py --embed_size 256 --enc_layers 2 --heads 8 --forward_expansion 2 --v_is_twolayer true -o baseline_fnl"
  "python train_eval.py --embed_size 128 --enc_layers 2 --heads 4 --forward_expansion 2 --v_is_twolayer true -o small_fnl"
  "python train_eval.py --embed_size 256 --enc_layers 4 --heads 8 --forward_expansion 2 --v_is_twolayer true -o deep_fnl"
  "python train_eval.py --embed_size 512 --enc_layers 2 --heads 8 --forward_expansion 4 --v_is_twolayer true -o wide_fnl"

  "python train_eval.py --embed_size 256 --enc_layers 2 --heads 8 --forward_expansion 2 --use_dynamic_clustering -o baseline_dc"
  "python train_eval.py --embed_size 128 --enc_layers 2 --heads 4 --forward_expansion 2 --use_dynamic_clustering -o small_dc"
  "python train_eval.py --embed_size 256 --enc_layers 4 --heads 8 --forward_expansion 2 --use_dynamic_clustering -o deep_dc"
  "python train_eval.py --embed_size 512 --enc_layers 2 --heads 8 --forward_expansion 4 --use_dynamic_clustering -o wide_dc"

  "python train_eval.py --embed_size 256 --enc_layers 2 --heads 8 --forward_expansion 2 --v_is_twolayer true --use_dynamic_clustering -o baseline_fnl_dc"
  "python train_eval.py --embed_size 128 --enc_layers 2 --heads 4 --forward_expansion 2 --v_is_twolayer true --use_dynamic_clustering -o small_fnl_dc"
  "python train_eval.py --embed_size 256 --enc_layers 4 --heads 8 --forward_expansion 2 --v_is_twolayer true --use_dynamic_clustering -o deep_fnl_dc"
  "python train_eval.py --embed_size 512 --enc_layers 2 --heads 8 --forward_expansion 4 --v_is_twolayer true --use_dynamic_clustering -o wide_fnl_dc"
)

CMD="${CMDS[$SLURM_ARRAY_TASK_ID]}"

echo "[$(date)] Running array task ${SLURM_ARRAY_TASK_ID} on host $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Command: $CMD"
echo

# Run each CMD with srun
srun $CMD
