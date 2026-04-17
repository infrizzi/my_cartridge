import os
import torch
import torch.distributed as dist
import time
from datetime import timedelta
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.train import TrainConfig, LossEvalConfig
from cartridges.datasets import TrainDataset, DataSource, LossEvalDataset
from cartridges.initialization.random import KVFromRandomVectors
import pydrantic

# Percorsi definiti al punto 1
TRAIN_PATH = "/work/tesi_lpaladino/outputs/processed_data/train_oppenheimer.parquet"
EVAL_PATH = "/work/tesi_lpaladino/outputs/processed_data/eval_oppenheimer.parquet"

def is_rank0():
    return not dist.is_initialized() or dist.get_rank() == 0

def print_rank0(*args, **kwargs):
    if is_rank0():
        print(*args, **kwargs)

def get_vram():
    return f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"

if __name__ == "__main__":

    if "RANK" in os.environ:
        dist.init_process_group("nccl")

    start_overall = time.time()
    print_rank0(f"--- Inizio Job: {time.ctime(start_overall)} | VRAM Iniziale: {get_vram()} ---")

    config = TrainConfig(
        model=HFModelConfig(
            pretrained_model_name_or_path="Qwen/Qwen3-4b",
            model_cls=FlexQwen3ForCausalLM,
        ),
        # Inizializzazione della cache apprendibile
        kv_cache_initializer=KVFromRandomVectors.Config(
            max_tokens=1024, 
            num_frozen_tokens=100
        ),
        
        lr=5e-4,
        epochs=50,
        global_batch_size=32,

        dataset=TrainDataset.Config(
            data_sources=[DataSource(path=TRAIN_PATH, type="local")],
            top_k_logits=20,
            packed_seq_length=1024,
            packing_mode="truncate",
        ),

        # --- SEZIONE EVALUATION --- 
        loss_eval_every_n_steps=100,
        loss_evals=[
            LossEvalConfig(
                dataset=LossEvalDataset.Config(
                    data_source=DataSource(path=EVAL_PATH, type="local"),
                    packed_seq_length=1024,
                ),
                name_for_wandb="oppenheimer_loss_eval",
            )
        ],

        save_every_n_steps=500,
        name="oppenheimer-distilled-v3",
        output_dir="/work/tesi_lpaladino/outputs/checkpoints",
    )

    print_rank0(f"--- VERIFICA CONFIGURAZIONE ---")
    print_rank0(f"Max Tokens: {config.kv_cache_initializer.max_tokens}") 
    print_rank0(f"Frozen Tokens richiesti: {config.kv_cache_initializer.num_frozen_tokens}")

    try:
        # Il wrapper CacheAndModel gestirà il forward con la block_mask
        pydrantic.main([config])
    finally:
        end_overall = time.time()
        duration = str(timedelta(seconds=int(end_overall - start_overall)))
        print_rank0(f"\n--- Job Terminato ---")
        print_rank0(f"Tempo totale impiegato: {duration}")
        print_rank0(f"VRAM finale occupata: {get_vram()}")
