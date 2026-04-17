#!/bin/bash
#SBATCH --job-name=trainCart
#SBATCH --output=logs/train/train_%j.out
#SBATCH --error=logs/train/train_%j.err
#SBATCH --account=tesi_lpaladino
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G

# Modules
module unload python/3.11.11-gcc-11.4.0
module load cuda/12.6.3-none-none

# Loading environment
source /work/tesi_lpaladino/cartridges_venv/bin/activate

# Paths
export CARTRIDGES_DIR="/homes/lpaladino/cartridges"
export CARTRIDGES_OUTPUT_DIR="/work/tesi_lpaladino/outputs"
export PYTHONPATH=$CARTRIDGES_DIR:$PYTHONPATH

# Training & GPU Optimization
export WANDB_MODE=offline
export TORCH_CUDA_ARCH_LIST="8.6;8.9"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Logs dir
mkdir -p logs/train
mkdir -p $CARTRIDGES_OUTPUT_DIR/checkpoints

echo "Inizio della fase di Training/Distillazione..."
echo "Utilizzo del dataset generato per Oppenheimer."

# Esecuzione dello script di training
torchrun --nproc_per_node=2 run_train.py

echo "Job di training completato con successo."
