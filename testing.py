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
    allocated = torch.cuda.memory_allocated() / 1024**3
    return f"{allocated:.2f} GB"

def cartridge_generate(model, tokenizer, cache, prompt, max_new_tokens=256, temp=0.7, rep_penalty=1.1):
    # 1. Reset della memoria temporanea (mantiene il Cartridge distillato)
    cache.clear()
    
    # 2. Tokenizzazione 1D
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"].squeeze(0) # (seq_len,)
    
    generated_ids_list = input_ids.tolist()
    seq_ids = torch.zeros_like(input_ids) 
    position_ids = torch.arange(len(input_ids), device="cuda")

    # 3. Loop di generazione
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                seq_ids=seq_ids,
                position_ids=position_ids,
                past_key_values=cache,
                mode="generate"
            )
            
            # --- FIX LOGITS: Selezioniamo correttamente l'ULTIMO token dell'ULTIMO batch ---
            # Shape attesa di outputs.logits: (1, seq_len, vocab_size)
            # Vogliamo (vocab_size,)
            logits = outputs.logits[0, -1, :] 
            
            # Applicazione Repetition Penalty
            # Usiamo un set per efficienza
            for token_id in set(generated_ids_list):
                if logits[token_id] > 0:
                    logits[token_id] /= rep_penalty
                else:
                    logits[token_id] *= rep_penalty
            
            # Sampling probabilistico
            probs = torch.softmax(logits / temp, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # (1,)
            
            # Aggiornamento liste e sequenze
            generated_ids_list.append(next_token.item())
            
            # Preparazione input per step successivo
            input_ids = next_token 
            seq_ids = torch.zeros(1, dtype=torch.long, device="cuda")
            position_ids = torch.tensor([len(generated_ids_list) - 1], device="cuda")

            if next_token.item() == tokenizer.eos_token_id:
                break
                
    # Decodifica escludendo il prompt
    full_text = tokenizer.decode(generated_ids_list, skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    return full_text[len(prompt_text):].strip()

def run_test():
    print(f"--- Inizio inference ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = FlexQwen3ForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    cache = TrainableCache.from_pretrained(CHECKPOINT_PATH).to("cuda")
    
    print(f"Cartridge pronto: {cache._num_frozen_tokens} Frozen, {cache._num_trainable_tokens} Trainable")

    test_prompts = [
        "Who was Lewis Strauss and what was his relationship with Robert Oppenheimer?",
        "Explain the importance of the Trinity test described in the movie.",
        "What does the final scene between Oppenheimer and Einstein by the lake represent?"
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[Test {i}] Domanda: {prompt}\n" + "-"*30)
        start_gen = time.time()
        risposta = cartridge_generate(model, tokenizer, cache, prompt)
        print(f"Risposta: {risposta}")
        print(f"Tempo: {time.time() - start_gen:.2f}s | VRAM: {get_vram_info()}")

if __name__ == "__main__":
    run_test()