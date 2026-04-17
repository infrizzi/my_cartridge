#!/bin/bash
#SBATCH --job-name=inferCart
#SBATCH --output=logs/test_standard/test_%j.out
#SBATCH --error=logs/test_standard/test_%j.err
#SBATCH --account=tesi_lpaladino
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=1:00:00
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G

# Modules
module unload python/3.11.11-gcc-11.4.0
module load cuda/12.6.3-none-none

export CARTRIDGES_DIR="/homes/lpaladino/cartridges"
export CARTRIDGES_OUTPUT_DIR="/work/tesi_lpaladino/outputs"
export PYTHONPATH="$CARTRIDGES_DIR:$PYTHONPATH"

# Loading environment
source /work/tesi_lpaladino/cartridges_venv/bin/activate

# Memory optm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Logs dir
mkdir -p logs/test_standard


echo "Inizio inference..."

# Esecuzione del test Python
python testing_standard.py
echo "Test concluso."