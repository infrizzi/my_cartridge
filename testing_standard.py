import torch
import time
import os
from datetime import timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configurazione percorsi
MODEL_ID = "Qwen/Qwen3-4b"
# Uso il percorso esatto che hai fornito nel log
SUBTITLES_PATH = "/work/tesi_lpaladino/documents/Oppenheimer282023%29.srt"

def get_vram_info():
    reserved = torch.cuda.memory_reserved() / 1024**3
    allocated = torch.cuda.memory_allocated() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return f"Allocata: {allocated:.2f}GB | Riservata: {reserved:.2f}GB | Picco: {peak:.2f}GB"

def run_classic_test():
    print(f"--- Caricamento Modello Standard ({MODEL_ID}) ---")
    start_load = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Carichiamo il modello standard di Hugging Face
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    # Caricamento del file sottotitoli 
    if os.path.exists(SUBTITLES_PATH):
        with open(SUBTITLES_PATH, 'r', encoding='utf-8') as f:
            subtitles_text = f.read()
    else:
        print(f"ERRORE: File non trovato in {SUBTITLES_PATH}")
        return

    end_load = time.time()
    print(f"Setup completato in {end_load - start_load:.2f} secondi.")
    print(f"VRAM post-caricamento: {get_vram_info()}\n")

    test_prompts = [
        "Who was Lewis Strauss and what was his relationship with Robert Oppenheimer?",
        "Explain the importance of the Trinity test described in the movie.",
        "What does the final scene between Oppenheimer and Einstein by the lake represent?"
    ]

    print("--- INIZIO GENERAZIONE STANDARD (No Cartridges) ---")
    start_total_gen = time.time()

    for i, user_query in enumerate(test_prompts, 1):
        full_prompt = f"Context from subtitles:\n{subtitles_text}\n\nQuestion: {user_query}\nAnswer:"
        
        print(f"\n[Test {i}] Domanda: {user_query}")
        print("-" * 30)
        
        torch.cuda.reset_peak_memory_stats()
        
        # Tokenizzazione del prompt massivo
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]
        
        start_gen = time.time()
        with torch.no_grad():
            # Usiamo la generate standard di HF
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                use_cache=True,
                temperature=0.7,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        duration = time.time() - start_gen
        
        # Decodifica solo dei nuovi token
        generated_text = tokenizer.decode(output_tokens[0][input_len:], skip_special_tokens=True)
        
        print(f"Risposta: {generated_text}")
        print(f"Tempo di generazione (Prefill + Gen): {duration:.2f} secondi")
        print(f"VRAM al picco: {get_vram_info()}")

    end_total = time.time()
    print(f"\n--- Riepilogo Finale Standard ---")
    print(f"Tempo totale: {str(timedelta(seconds=int(end_total - start_total_gen)))}")

if __name__ == "__main__":
    run_classic_test()