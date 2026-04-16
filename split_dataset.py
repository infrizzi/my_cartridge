import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Definiamo i percorsi
original_path = "/work/tesi_lpaladino/outputs/2026-04-15-16-30-44-synthesis/cccb0dd8-4041-4c9d-95f7-0b548d1e3f22/artifact/dataset.parquet"
output_dir = "/work/tesi_lpaladino/outputs/processed_data"

# Crea la cartella di output se non esiste
os.makedirs(output_dir, exist_ok=True)

# Carica il dataset
print("Caricamento dataset originale...")
df = pd.read_parquet(original_path)

# Split 90% train, 10% eval
train_df, eval_df = train_test_split(df, test_size=0.10, random_state=42)

# Salva
train_path = os.path.join(output_dir, "train_oppenheimer.parquet")
eval_path = os.path.join(output_dir, "eval_oppenheimer.parquet")

train_df.to_parquet(train_path)
eval_df.to_parquet(eval_path)

print(f"✅ Split completato con successo!")
print(f"📂 Train: {train_path} ({len(train_df)} campioni)")
print(f"📂 Eval: {eval_path} ({len(eval_df)} campioni)")
