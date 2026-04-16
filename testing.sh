#!/bin/bash
#SBATCH --job-name=inferCart
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --account=tesi_lpaladino
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=1:00:00
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G|gpu_RTX6000_24G|gpu_RTX_A5000_24G

# Modules
module unload python/3.11.11-gcc-11.4.0
module load cuda/12.6.3-none-none

export CARTRIDGES_DIR="/homes/lpaladino/cartridges"
export PYTHONPATH="$CARTRIDGES_DIR:$PYTHONPATH"

# Loading environment
source /work/tesi_lpaladino/cartridges_venv/bin/activate

# Memory optm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Logs dir
mkdir -p logs

# Monitor VRAM
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv > logs/gpu_test_stats_${SLURM_JOB_ID}.csv &
MONITOR_PID=$!

echo "Inizio inference..."

# Esecuzione del test Python
python testing.py

# Pulizia: uccidiamo il server alla fine del test
echo "Test concluso."
kill $MONITOR_PID