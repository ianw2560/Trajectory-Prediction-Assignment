#!/bin/bash

#SBATCH --job-name=traj_analysis
#SBATCH --gres=gpu:nvidia_h100_pcie:1    
#SBATCH --cpus-per-task=16        # tweak as needed
#SBATCH --mem=32G                # tweak as needed
#SBATCH --time=3:00:00          # tweak as needed
#SBATCH --array=0-8              # 9 jobs: indices 0..8
#SBATCH -o logs/traj_analysis_%A_%a.out  # %A = array job id, %a = task id


# conda activate your_env

# List of commands: one per array index
CMDS=(
  "python train_eval.py --embed_size 256 --enc_layers 2 --heads 8 --forward_expansion 2 -o arch_baseline"
  "python train_eval.py --embed_size 128 --enc_layers 2 --heads 4 --forward_expansion 2 -o arch_small"
  "python train_eval.py --embed_size 256 --enc_layers 4 --heads 8 --forward_expansion 2 -o arch_deep"
  "python train_eval.py --embed_size 512 --enc_layers 2 --heads 8 --forward_expansion 4 -o arch_wide"

  "python train_eval.py --v_is_twolayer true -o v_is_two_layer"

  "python train_eval.py --embed_size 256 --enc_layers 2 --heads 8 --forward_expansion 2 --use_dynamic_clustering -o use_dynamic_clustering_baseline"
  "python train_eval.py --embed_size 128 --enc_layers 2 --heads 4 --forward_expansion 2 --use_dynamic_clustering -o use_dynamic_clustering_small"
  "python train_eval.py --embed_size 256 --enc_layers 4 --heads 8 --forward_expansion 2 --use_dynamic_clustering -o use_dynamic_clustering_deep"
  "python train_eval.py --embed_size 512 --enc_layers 2 --heads 8 --forward_expansion 4 --use_dynamic_clustering -o use_dynamic_clustering_wide"
)

CMD="${CMDS[$SLURM_ARRAY_TASK_ID]}"

echo "[$(date)] Running array task ${SLURM_ARRAY_TASK_ID} on host $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Command: $CMD"
echo

# srun helps SLURM track the task properly
srun $CMD
