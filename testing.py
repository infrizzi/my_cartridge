import torch
import time
from datetime import timedelta
from transformers import AutoTokenizer
from cartridges.models import FlexQwen3ForCausalLM
from cartridges.cache import TrainableCache

# Configurazione percorsi
MODEL_ID = "Qwen/Qwen3-4b"
CHECKPOINT_PATH = "/work/tesi_lpaladino/outputs/checkpoints/2026-04-16-14-32-16-run_train/4551b584-fe03-4a73-81ff-89204d6ce629/cache-step32.pt"

def run_test():
    print(f"--- Caricamento Modello e Tokenizer ({MODEL_ID}) ---")
    start_load = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = FlexQwen3ForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda"
    )
    
    # Carichiamo il tuo Cartridge specifico
    print(f"--- Caricamento Cartridge: {CHECKPOINT_PATH.split('/')[-1]} ---")
    cache = TrainableCache.from_pretrained(CHECKPOINT_PATH).to("cuda")
    cache._num_frozen_tokens = 100
    cache._num_trainable_tokens = 924

    print(f"Configurazione Cache: Frozen={cache._num_frozen_tokens}, Trainable={cache._num_trainable_tokens}")
    
    end_load = time.time()
    print(f"Setup completato in {end_load - start_load:.2f} secondi.\n")

    test_prompts = [
        "Chi è Lewis Strauss e qual è il suo rapporto con Robert Oppenheimer?",
        "Spiegami l'importanza del test Trinity descritto nel film.",
        "Cosa rappresenta la scena finale tra Oppenheimer ed Einstein vicino al laghetto?"
    ]

    print("--- INIZIO GENERAZIONE ---")
    start_total_gen = time.time()

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[Test {i}] Domanda: {prompt}")
        print("-" * 30)
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        generated_ids = inputs["input_ids"]
        
        start_gen = time.time()
        with torch.no_grad():
            for _ in range(256): # max_new_tokens
                # Passiamo la cache direttamente al forward del modello
                outputs = model(input_ids=generated_ids, cache=cache)
                
                # Prendiamo i logits dell'ultimo token generato
                next_token_logits = outputs.logits[:, -1, :]
                
                # Greedy Decoding: prendiamo il token con probabilità massima
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                # Concateniamo il nuovo token alla sequenza
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                # Controllo fine della generazione (EOS token)
                if next_token.item() == tokenizer.eos_token_id:
                    break
        duration = time.time() - start_gen
        
        risposta = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # Rimuoviamo il prompt dalla risposta per pulizia
        risposta_pulita = risposta[len(prompt):].strip()
        
        print(f"Risposta: {risposta_pulita}")
        print(f"Tempo di generazione: {duration:.2f} secondi")

    end_total = time.time()
    print(f"\n--- Riepilogo Finale ---")
    print(f"Tempo di generazione totale: {str(timedelta(seconds=int(end_total - start_total_gen)))}")

if __name__ == "__main__":
    run_test()