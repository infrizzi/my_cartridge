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
    
    end_load = time.time()
    print(f"Setup completato in {end_load - start_load:.2f} secondi.\n")

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
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        generated_ids = inputs["input_ids"]
        input_ids = inputs["input_ids"].squeeze(0)
        seq_ids = torch.zeros_like(input_ids)
        position_ids = torch.arange(len(input_ids)).to("cuda")

        max_new_tokens = 256
        
        start_gen = time.time()
        with torch.no_grad():
            for i in range(max_new_tokens):
                # CHIAMATA CORRETTA:
                # 1. Usiamo i nomi dei parametri della classe (past_key_values invece di cache)
                # 2. Passiamo seq_ids e position_ids che sono obbligatori
                # 3. Impostiamo mode="generate" come previsto dalla classe Qwen3Batch
                outputs = model(
                    input_ids=input_ids,
                    seq_ids=seq_ids,
                    position_ids=position_ids,
                    past_key_values=cache,
                    mode="generate"
                )
                
                # Prendiamo i logits dell'ULTIMO token
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                # Aggiorniamo la sequenza totale per la decodifica finale
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                # --- LOGICA KV-CACHE ---
                # Per il passo successivo, passiamo SOLO l'ultimo token generato.
                # La cache (past_key_values) tiene già a mente tutto il passato.
                input_ids = next_token.squeeze(0)
                seq_ids = torch.zeros_like(input_ids)
                # La posizione del nuovo token è la lunghezza attuale della sequenza
                position_ids = torch.tensor([len(generated_ids[0]) - 1]).to("cuda")

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