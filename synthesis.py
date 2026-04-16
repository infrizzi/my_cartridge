from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.data.resources import TextFileResource
from cartridges.data.chunkers import TokenChunker
from cartridges.clients.tokasaurus import TokasaurusClient
import pydrantic
import time
from datetime import timedelta

# 1. Configura il client che parlerà con il server Tokasaurus
client_config = TokasaurusClient.Config(
    url="http://127.0.0.1:10555", # Useremo questa porta nel server
    model_name="Qwen/Qwen3-4b"
)

# 2. Configura la risorsa (il tuo file .srt)
resource_config = TextFileResource.Config(
    path="/work/tesi_lpaladino/documents/Oppenheimer282023%29.srt",
    seed_prompts=["summarization", "question", "structuring"],
    chunker=TokenChunker.Config(
        tokenizer="Qwen/Qwen3-4b",
        min_tokens_per_chunk=512,
        max_tokens_per_chunk=1024,
    ),
)

# 3. Configurazione finale
config = SynthesizeConfig(
    synthesizer=SelfStudySynthesizer.Config(
        client=client_config,
        resources=[resource_config],
	tools=[],
    ),
    batch_size=32,
    max_num_batches_in_parallel=4,
    num_samples=1024, # Numero di conversazioni da generare
    name="cartridge-srt-tutorial",
)

if __name__ == "__main__":
    start_time = time.time()
    print(f"--- Inizio Sintesi: {time.ctime(start_time)} ---")

    try:
        pydrantic.main([config])
    finally:
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Formattiamo il tempo in modo leggibile (ore:minuti:secondi)
        readable_time = str(timedelta(seconds=int(total_duration)))
        
        # Calcoliamo la velocità media
        avg_speed = total_duration / config.num_samples if config.num_samples > 0 else 0

        print("\n" + "="*50)
        print(f"SINTESI COMPLETATA")
        print(f"Tempo totale: {readable_time}")
        print(f"Velocità media: {avg_speed:.2f} secondi/campione")
        print(f"Campioni totali: {config.num_samples}")
        print("="*50)
