import requests
import json
import time
from datetime import timedelta

# URL del server Tokasaurus
URL = "http://127.0.0.1:10210/v1/chat/completions"

# Percorso assoluto del tuo miglior checkpoint
CARTRIDGE_PATH = "/work/tesi_lpaladino/outputs/checkpoints/2026-04-16-14-32-16-run_train/4551b584-fe03-4a73-81ff-89204d6ce629/cache-step32.pt"

def ask_cartridge(prompt):
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.4, # Bassa per risposte più precise e meno allucinate
        "cartridges": [{
            "id": CARTRIDGE_PATH,
            "source": "local",
            "force_redownload": False
        }]
    }
    
    start_req = time.time()
    try:
        response = requests.post(URL, json=payload)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        duration = time.time() - start_req
        return content, duration
    except Exception as e:
        return f"Errore durante la richiesta: {e}", 0

if __name__ == "__main__":
    test_prompts = [
        "Chi è Lewis Strauss e qual è il suo rapporto con Robert Oppenheimer?",
        "Spiegami l'importanza del test Trinity descritto nel film.",
        "Cosa rappresenta la scena finale tra Oppenheimer ed Einstein vicino al laghetto?"
    ]
    start_total = time.time()

    print("\n--- TEST DEL CARTRIDGE OPPENHEIMER ---")
    for i, p in enumerate(test_prompts, 1):
        print(f"\n[Test {i}] Domanda: {p}")
        print("-" * 30)
        risposta, durata = ask_cartridge(p)
        print(f"Risposta: {risposta}\n")
        print(f"Tempo di risposta: {durata:.2f} secondi")

    end_total = time.time()
    total_duration = str(timedelta(seconds=int(end_total - start_total)))
    print(f"Tempo totale di esecuzione test: {total_duration}")