import pandas as pd
import os

# Percorsi
original_path = "/work/tesi_lpaladino/outputs/2026-04-15-16-30-44-synthesis/cccb0dd8-4041-4c9d-95f7-0b548d1e3f22/artifact/dataset.parquet"
output_dir = "/work/tesi_lpaladino/outputs/processed_data"
os.makedirs(output_dir, exist_ok=True)

# Caricamento e mischiamento (shuffling)
print("Caricamento dataset...")
df = pd.read_parquet(original_path)
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Calcolo del punto di split (90%)
split_point = int(len(df_shuffled) * 0.9)

# Divisione
train_df = df_shuffled.iloc[:split_point]
eval_df = df_shuffled.iloc[split_point:]

# Salvataggio
train_path = os.path.join(output_dir, "train_oppenheimer.parquet")
eval_path = os.path.join(output_dir, "eval_oppenheimer.parquet")

train_df.to_parquet(train_path)
eval_df.to_parquet(eval_path)

print(f"✅ Split completato (metodo Pandas)!")
print(f"📂 Train: {len(train_df)} righe | Eval: {len(eval_df)} righe")