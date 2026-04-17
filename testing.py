import torch
import time
from datetime import timedelta
from transformers import AutoTokenizer
from cartridges.models import FlexQwen3ForCausalLM
from cartridges.cache import TrainableCache

MODEL_ID = "Qwen/Qwen3-4b"
CHECKPOINT_PATH = "/work/tesi_lpaladino/outputs/checkpoints/2026-04-17-10-55-15-run_train/bc24dcb7-25fa-4aba-a9e7-c638bfeb5d77/cache-step550.pt"

def get_vram_info():
    """Monitoraggio della VRAM allocata."""
    allocated = torch.cuda.memory_allocated() / 1024**3
    return f"Allocata: {allocated:.2f} GB"

def run_fixed_test():
    print(f"--- Caricamento Modello e Tokenizer ({MODEL_ID}) ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = FlexQwen3ForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    
    # Caricamento del Cartridge [cite: 15, 51]
    print(f"--- Caricamento Cartridge: cache-step550.pt ---")
    cache = TrainableCache.from_pretrained(CHECKPOINT_PATH).to("cuda")
    
    # Verifica strutturale (100 frozen, 924 trainable) 
    n_frozen = cache._num_frozen_tokens
    n_trainable = cache._num_trainable_tokens
    print(f"--- Verifica Struttura: {n_frozen} Frozen | {n_trainable} Trainable ---")
    assert n_frozen == 100, f"Errore: trovati {n_frozen} token frozen invece di 100"

    test_prompts = [
        "Who was Lewis Strauss and what was his relationship with Robert Oppenheimer?",
        "Explain the importance of the Trinity test described in the movie.",
        "What does the final scene between Oppenheimer and Einstein by the lake represent?"
    ]

    # Parametri di generazione per contrastare la degenerazione del testo [cite: 67]
    MAX_NEW_TOKENS = 256
    TEMPERATURE = 0.7
    REPETITION_PENALTY = 1.1

    print("--- INIZIO GENERAZIONE ---")
    start_total_gen = time.time()

    for i, prompt in enumerate(test_prompts, 1):
        # Reset della memoria temporanea (mantiene il Cartridge intatto) 
        cache.clear() 
        
        print(f"\n[Test {i}] Domanda: {prompt}\n" + "-"*30)
        
        # Tokenizzazione iniziale
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        generated_ids = inputs["input_ids"].clone()
        
        # --- CONFIGURAZIONE INIZIALE TENSORI ---
        # input_ids: deve essere 2D (batch, seq_len)
        input_ids = inputs["input_ids"] 
        # seq_ids: deve essere 1D per la block_mask interna [cite: 32, 89]
        seq_ids = torch.zeros(input_ids.shape[1], dtype=torch.long, device="cuda")
        # position_ids: deve essere 2D (batch, seq_len) per allinearsi al RoPE 
        position_ids = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0)

        start_gen = time.time()
        with torch.no_grad():
            for _ in range(MAX_NEW_TOKENS):
                # Forward pass con modalità generazione [cite: 64, 126]
                outputs = model(
                    input_ids=input_ids,
                    seq_ids=seq_ids,
                    position_ids=position_ids,
                    past_key_values=cache,
                    mode="generate"
                )
                
                # Estrazione logits dell'ultimo token
                next_token_logits = outputs.logits[:, -1, :]
                
                # Repetition Penalty
                for token_id in set(generated_ids[0].tolist()):
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= REPETITION_PENALTY
                    else:
                        next_token_logits[0, token_id] *= REPETITION_PENALTY

                # Sampling probabilistico
                probs = torch.softmax(next_token_logits / TEMPERATURE, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1) # (1, 1)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                # --- AGGIORNAMENTO AUTOREGRESSIVO (Fix per RoPE) ---
                input_ids = next_token # Mantiene (1, 1)
                seq_ids = torch.zeros(1, dtype=torch.long, device="cuda") # 1D
                # La posizione deve riflettere l'indice assoluto nella sequenza totale
                position_ids = torch.tensor([[generated_ids.shape[1] - 1]], device="cuda") # 2D

                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        duration = time.time() - start_gen
        # Decodifica escludendo il prompt
        risposta = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"Risposta: {risposta.strip()}\nTempo: {duration:.2f}s | {get_vram_info()}")

    end_total = time.time()
    print(f"\n--- Riepilogo Finale ---")
    print(f"Tempo totale: {str(timedelta(seconds=int(end_total - start_total_gen)))}")

if __name__ == "__main__":
    run_fixed_test()