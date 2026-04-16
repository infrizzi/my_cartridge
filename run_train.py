import torch
import time
from datetime import timedelta
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.train import TrainConfig, GenerationEvalConfig, LossEvalConfig
from cartridges.datasets import TrainDataset, DataSource, GenerateEvalDataset, LossEvalDataset
from cartridges.initialization.random import KVFromRandomVectors
import pydrantic

# Percorsi definiti al punto 1
TRAIN_PATH = "/work/tesi_lpaladino/outputs/processed_data/train_oppenheimer.parquet"
EVAL_PATH = "/work/tesi_lpaladino/outputs/processed_data/eval_oppenheimer.parquet"

def get_vram():
    return f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"

if __name__ == "__main__":
    start_overall = time.time()
    print(f"--- Inizio Job: {time.ctime(start_overall)} | VRAM Iniziale: {get_vram()} ---")

    config = TrainConfig(
        model=HFModelConfig(
            pretrained_model_name_or_path="Qwen/Qwen3-4b",
            model_cls=FlexQwen3ForCausalLM,
        ),
        # Inizializzazione della cache apprendibile [cite: 4, 5]
        kv_cache_initializer=KVFromRandomVectors.Config(
            max_tokens=1024, 
            num_frozen_tokens=100 # Come visto nei tuoi log precedenti [cite: 7]
        ),
        
        lr=2e-2,
        epochs=3,
        global_batch_size=32,

        dataset=TrainDataset.Config(
            data_sources=[DataSource(path=TRAIN_PATH, type="local")],
            top_k_logits=20,
            packed_seq_length=2048,
            packing_mode="truncate",
        ),

        # --- SEZIONE EVALUATION --- 
        loss_eval_every_n_steps=100,
        loss_evals=[
            LossEvalConfig(
                dataset=LossEvalDataset.Config(
                    data_source=DataSource(path=EVAL_PATH, type="local"),
                    packed_seq_length=2048,
                ),
                name_for_wandb="oppenheimer_loss_eval",
            )
        ],

        generate_eval_every_n_steps=100,
        generate_evals=[
            GenerationEvalConfig(
                dataset=GenerateEvalDataset.Config(
                    data_source=DataSource(path=EVAL_PATH, type="local"),
                    max_samples=10, 
                ),
                num_samples=1,
                temperature=0.3, # Risposte più deterministiche
                name_for_wandb="oppenheimer_generation_test",
            )
        ],

        save_every_n_steps=200,
        name="oppenheimer-distilled-v2",
        output_dir="/work/tesi_lpaladino/outputs/checkpoints",
    )

    try:
        # Il wrapper CacheAndModel gestirà il forward con la block_mask [cite: 60, 87]
        pydrantic.main([config])
    finally:
        end_overall = time.time()
        duration = str(timedelta(seconds=int(end_overall - start_overall)))
        print(f"\n--- Job Terminato ---")
        print(f"Tempo totale impiegato: {duration}")
        print(f"VRAM finale occupata: {get_vram()}")
