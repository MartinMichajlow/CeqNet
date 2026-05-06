#!/usr/bin/env bash
#SBATCH --job-name=ceqnet_qm9
#SBATCH --partition=gpu-5h
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=1-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00

mkdir -p logs

CONTAINER=/path/to/container/base.sif
SCRIPT=$(dirname "$0")/train.py

apptainer exec --nv "$CONTAINER" python "$SCRIPT" --run_idx "$SLURM_ARRAY_TASK_ID"
