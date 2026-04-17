import torch
import time
from datetime import timedelta
from transformers import AutoTokenizer
from cartridges.models import FlexQwen3ForCausalLM
from cartridges.cache import TrainableCache

MODEL_ID = "Qwen/Qwen3-4b"
CHECKPOINT_PATH = "/work/tesi_lpaladino/outputs/checkpoints/2026-04-17-10-55-15-run_train/bc24dcb7-25fa-4aba-a9e7-c638bfeb5d77/cache-step550.pt"

def get_vram_info():
    allocated = torch.cuda.memory_allocated() / 1024**3
    return f"Allocata: {allocated:.2f} GB"

def run_fixed_test():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = FlexQwen3ForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    cache = TrainableCache.from_pretrained(CHECKPOINT_PATH).to("cuda")
    
    # Verifica strutturale 
    print(f"Cartridge pronto: {cache._num_frozen_tokens} Frozen, {cache._num_trainable_tokens} Trainable")

    test_prompts = [
        "Who was Lewis Strauss and what was his relationship with Robert Oppenheimer?",
        "Explain the importance of the Trinity test described in the movie.",
        "What does the final scene between Oppenheimer and Einstein by the lake represent?"
    ]

    # Parametri 
    MAX_NEW_TOKENS = 256
    TEMPERATURE = 0.7
    REPETITION_PENALTY = 1.1
    CARTRIDGE_LEN = 1024 

    for i, prompt in enumerate(test_prompts, 1):
        # --- FIX 1: Pulizia della cache per ogni nuova domanda [cite: 120] ---
        cache.clear() 
        
        print(f"\n[Test {i}] Domanda: {prompt}\n" + "-"*30)
        
        # --- FIX 2: Mantenere la dimensione Batch (2D) ---
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"] # (1, L)
        generated_ids = input_ids.clone()
        
        # Inizializzazione seq_ids e position_ids con offset
        seq_ids = torch.zeros(input_ids.shape, dtype=torch.long, device="cuda")
        position_ids = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0) + CARTRIDGE_LEN

        with torch.no_grad():
            for _ in range(MAX_NEW_TOKENS):
                outputs = model(
                    input_ids=input_ids,
                    seq_ids=seq_ids,
                    position_ids=position_ids,
                    past_key_values=cache,
                    mode="generate"
                )
                
                next_token_logits = outputs.logits[:, -1, :]
                
                # Repetition Penalty
                for token_id in set(generated_ids[0].tolist()):
                    next_token_logits[0, token_id] /= REPETITION_PENALTY

                # Sampling
                probs = torch.softmax(next_token_logits / TEMPERATURE, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                # Update per token successivo (2D)
                input_ids = next_token
                seq_ids = torch.zeros((1, 1), dtype=torch.long, device="cuda")
                position_ids = torch.tensor([[generated_ids.shape[1] - 1 + CARTRIDGE_LEN]], device="cuda")

                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        risposta = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"Risposta: {risposta.strip()}\n{get_vram_info()}")

if __name__ == "__main__":
    run_fixed_test()