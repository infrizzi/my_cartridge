import torch
import time
from datetime import timedelta
from transformers import AutoTokenizer
from cartridges.models import FlexQwen3ForCausalLM
from cartridges.cache import AttnConfig, TrainableCache

# Configurazione percorsi
MODEL_ID = "Qwen/Qwen3-4b"
SUBTITLES_PATH = "/work/tesi_lpaladino/documents/Oppenheimer282023%29.srt"

def get_vram_info():
    reserved = torch.cuda.memory_reserved() / 1024**3
    allocated = torch.cuda.memory_allocated() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return f"Allocata: {allocated:.2f}GB | Riservata: {reserved:.2f}GB | Picco: {peak:.2f}GB"

def run_icl_test():
    print(f"--- Caricamento Modello e Tokenizer ({MODEL_ID}) ---")
    start_load = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = FlexQwen3ForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda"
    )

    qwen_attn_config = AttnConfig(
        n_layers=37,    # 
        n_heads=8,      # 
        head_dim=128    # 
    )

    empty_cache = TrainableCache(
        config=qwen_attn_config,
        init_keys=None,
        init_values=None,
        num_frozen_tokens=0
        ).to("cuda")
    
    # Caricamento del file sottotitoli 
    with open(SUBTITLES_PATH, 'r', encoding='utf-8') as f:
        subtitles_text = f.read()
    
    end_load = time.time()
    print(f"Setup completato in {end_load - start_load:.2f} secondi.")
    print(f"VRAM dopo caricamento modello: {get_vram_info()}\n")

    test_prompts = [
        "Who was Lewis Strauss and what was his relationship with Robert Oppenheimer?",
        "Explain the importance of the Trinity test described in the movie.",
        "What does the final scene between Oppenheimer and Einstein by the lake represent?"
    ]

    print("--- INIZIO GENERAZIONE ICL (Full Context) ---")
    start_total_gen = time.time()

    for i, user_query in enumerate(test_prompts, 1):
        # Costruiamo il prompt ICL completo
        full_prompt = f"Context from Oppenheimer subtitles:\n{subtitles_text}\n\nQuestion: {user_query}\nAnswer:"
        
        print(f"\n[Test {i}] Domanda: {user_query}")
        print("-" * 30)
        
        # Reset picco VRAM per ogni test
        torch.cuda.reset_peak_memory_stats()
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        generated_ids = inputs["input_ids"]
        input_ids = inputs["input_ids"]
        
        # Inizializzazione parametri obbligatori per FlexQwen3
        seq_ids = torch.zeros_like(input_ids)
        position_ids = torch.arange(input_ids.size(1)).unsqueeze(0).to("cuda")
        
        # past_key_values inizia None per il prefill massivo
        past_key_values = empty_cache
        max_new_tokens = 256
        
        start_gen = time.time()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Il modello gestisce il prefill al primo step e l'append nei successivi
                outputs = model(
                    input_ids=input_ids,
                    seq_ids=seq_ids,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    mode="generate" # Fondamentale per attivare la logica corretta
                )
                
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                # Prepariamo l'input per il token successivo (Autoregressive)
                input_ids = next_token
                seq_ids = torch.zeros_like(input_ids)
                position_ids = torch.tensor([[generated_ids.size(1) - 1]], device="cuda")

                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        gen_duration = time.time() - start_gen
        risposta = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        risposta_pulita = risposta[len(full_prompt):].strip()
        
        print(f"Risposta: {risposta_pulita}")
        print(f"Tempo di generazione: {gen_duration:.2f} secondi")
        print(f"VRAM al picco durante test: {get_vram_info()}")

    end_total = time.time()
    print(f"\n--- Riepilogo Finale ICL ---")
    print(f"Tempo di generazione totale: {str(timedelta(seconds=int(end_total - start_total_gen)))}")

if __name__ == "__main__":
    run_icl_test()