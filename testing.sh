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

# Loading environment
source /work/tesi_lpaladino/cartridges_venv/bin/activate

export HOST=0.0.0.0
export TORCH_CUDA_ARCH_LIST="8.6;8.9"

# Memory optm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Logs dir
mkdir -p logs

# Monitor VRAM
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv > logs/gpu_test_stats_${SLURM_JOB_ID}.csv &
MONITOR_PID=$!

echo "Avvio del server Tokasaurus..."
tksrs model=Qwen/Qwen3-4b kv_cache_num_tokens='(32 * 1024)' port=10210 &

echo "Verifica disponibilità server sulla porta 10210..."
until curl -s http://127.0.0.1:10210/v1/chat/completions > /dev/null; do
  echo "Il server non risponde ancora... attendo 15 secondi"
  sleep 15
done
echo "Server pronto! Eseguo testing"

# Esecuzione del test Python
python testing.py

# Pulizia: uccidiamo il server alla fine del test
echo "Test concluso. Arresto del server..."
kill %1
kill $MONITOR_PID