import requests
import json

# URL del server Tokasaurus (porta standard 10210)
URL = "http://127.0.0.1:10210/v1/cartridge/chat/completions"

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
    
    try:
        response = requests.post(URL, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Errore durante la richiesta: {e}"

if __name__ == "__main__":
    test_prompts = [
        "Chi è Lewis Strauss e qual è il suo rapporto con Robert Oppenheimer?",
        "Spiegami l'importanza del test Trinity descritto nel film.",
        "Cosa rappresenta la scena finale tra Oppenheimer ed Einstein vicino al laghetto?"
    ]

    print("\n--- TEST DEL CARTRIDGE OPPENHEIMER ---")
    for i, p in enumerate(test_prompts, 1):
        print(f"\n[Test {i}] Domanda: {p}")
        print("-" * 30)
        risposta = ask_cartridge(p)
        print(f"Risposta: {risposta}\n")