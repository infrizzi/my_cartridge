import torch
import time
import os
from datetime import timedelta
from transformers import AutoTokenizer
from cartridges.models import FlexQwen3ForCausalLM
from cartridges.cache import TrainableCache

# --- CONFIGURAZIONE ---
MODEL_ID = "Qwen/Qwen3-4b"
CHECKPOINT_PATH = "/work/tesi_lpaladino/outputs/checkpoints/2026-04-17-10-55-15-run_train/bc24dcb7-25fa-4aba-a9e7-c638bfeb5d77/cache-step550.pt"

def get_vram_info():
    """Restituisce il consumo attuale di VRAM."""
    allocated = torch.cuda.memory_allocated() / 1024**3
    return f"{allocated:.2f} GB"

def cartridge_generate(model, tokenizer, cache, prompt, max_new_tokens=256, temp=0.7, rep_penalty=1.1):
    """
    Esegue una generazione fluida utilizzando il Cartridge.
    Gestisce internamente la pulizia della cache e la logica 1D per evitare errori RoPE.
    """
    # 1. Reset della memoria temporanea (mantiene il Cartridge distillato) [cite: 322]
    cache.clear()
    
    # 2. Tokenizzazione 1D (Squeeziamo per rimuovere la dimensione batch)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"].squeeze(0) # (seq_len,)
    
    generated_ids = input_ids.clone()
    # seq_ids mappati a 0 per il prompt (il Cartridge usa -1 internamente) [cite: 9, 89]
    seq_ids = torch.zeros_like(input_ids) 
    # position_ids lineari (la logica interna gestirà l'integrazione con la cache) [cite: 108]
    position_ids = torch.arange(len(input_ids), device="cuda")

    # 3. Loop di generazione autoregressiva
    for _ in range(max_new_tokens):
        with torch.no_grad():
            # Passaggio al modello FlexQwen3 [cite: 60, 106]
            outputs = model(
                input_ids=input_ids,
                seq_ids=seq_ids,
                position_ids=position_ids,
                past_key_values=cache,
                mode="generate"
            )
            
            # Estrazione dei logits dell'ultimo token (formato 1D: vocab_size)
            logits = outputs.logits[-1, :] 
            
            # Applicazione Repetition Penalty manuale
            for token_id in set(generated_ids.tolist()):
                if logits[token_id] > 0:
                    logits[token_id] /= rep_penalty
                else:
                    logits[token_id] *= rep_penalty
            
            # Sampling probabilistico con Temperatura
            probs = torch.softmax(logits / temp, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # (1,)
            
            # Aggiornamento della sequenza totale per la decodifica
            generated_ids = torch.cat([generated_ids, next_token])
            
            # --- AGGIORNAMENTO PER IL PROSSIMO STEP (Sempre 1D) ---
            input_ids = next_token # (1,)
            seq_ids = torch.zeros(1, dtype=torch.long, device="cuda")
            # La nuova posizione è semplicemente la lunghezza attuale della sequenza generata
            position_ids = torch.tensor([len(generated_ids) - 1], device="cuda")

            if next_token.item() == tokenizer.eos_token_id:
                break
                
    # Decodifica escludendo il prompt iniziale
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return full_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()

def run_test():
    print(f"--- Setup Ambiente HPC ---")
    start_load = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = FlexQwen3ForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda"
    )
    
    # Caricamento del Cartridge addestrato [cite: 15, 51]
    print(f"Caricamento Checkpoint: {CHECKPOINT_PATH.split('/')[-1]}")
    cache = TrainableCache.from_pretrained(CHECKPOINT_PATH).to("cuda")
    
    # VERIFICA STRUTTURALE (100 frozen, 924 trainable) [cite: 7, 260]
    n_frozen = cache._num_frozen_tokens
    n_trainable = cache._num_trainable_tokens
    print(f"--- Verifica Cartridge: {n_frozen} Frozen | {n_trainable} Trainable ---")
    
    if n_frozen != 100:
        print(f"ATTENZIONE: Trovati {n_frozen} token frozen. Verifica cache.py per il bug .size(1) vs .size(2)")

    end_load = time.time()
    print(f"Setup completato in {end_load - start_load:.2f}s | VRAM: {get_vram_info()}\n")

    test_prompts = [
        "Who was Lewis Strauss and what was his relationship with Robert Oppenheimer?",
        "Explain the importance of the Trinity test described in the movie.",
        "What does the final scene between Oppenheimer and Einstein by the lake represent?"
    ]

    print("--- INIZIO GENERAZIONE ---")
    start_total_gen = time.time()

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[Test {i}] Domanda: {prompt}")
        print("-" * 30)
        
        start_gen = time.time()
        risposta = cartridge_generate(model, tokenizer, cache, prompt)
        duration = time.time() - start_gen
        
        print(f"Risposta: {risposta}")
        print(f"Tempo: {duration:.2f}s | VRAM: {get_vram_info()}")

    end_total = time.time()
    print(f"\n--- Riepilogo Finale ---")
    print(f"Tempo totale di generazione: {str(timedelta(seconds=int(end_total - start_total_gen)))}")
    print(f"VRAM Picco: {get_vram_info()} (Baseline standard era >32GB)")

if __name__ == "__main__":
    run_test()