#!/bin/bash
#SBATCH --job-name=selfstudyCart
#SBATCH --output=logs/selfstudy_%j.out
#SBATCH --error=logs/selfstudy_%j.err
#SBATCH --account=tesi_lpaladino
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2:00:00
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

export HOST=0.0.0.0
export TORCH_CUDA_ARCH_LIST="8.6;8.9"

# Memory optm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Logs dir
mkdir -p logs
mkdir -p $CARTRIDGES_OUTPUT_DIR

# Monitoring VRAM
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 60 > logs/gpu_stats_${SLURM_JOB_ID}.csv &
MONITOR_PID=$!

echo "Avvio del server Tokasaurus..."
tksrs model=Qwen/Qwen3-4b kv_cache_num_tokens='(128 * 1024)' port=10555 max_topk_logprobs=20 &

echo "Verifica disponibilità server sulla porta 10555..."
until curl -s http://127.0.0.1:10555/v1/models > /dev/null; do
  echo "Il server non risponde ancora... attendo 15 secondi"
  sleep 15
done
echo "Server pronto! Lancio la sintesi..."

export WANDB_MODE=offline
python synthesis.py

kill %1
kill $MONITOR_PID

echo "Job completato con successo."
