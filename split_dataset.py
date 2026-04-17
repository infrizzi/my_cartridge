import pandas as pd
import os

# Percorsi
original_path = "/work/tesi_lpaladino/outputs/2026-04-17-15-04-03-synthesis/a179f11d-d39c-4c0b-b89a-8fdc85f469ee/artifact/dataset.parquet"
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
train_path = os.path.join(output_dir, "train_oppenheimer_v4.parquet")
eval_path = os.path.join(output_dir, "eval_oppenheimer_v4.parquet")

train_df.to_parquet(train_path)
eval_df.to_parquet(eval_path)

print(f"✅ Split completato!")
print(f"📂 Train: {len(train_df)} righe | Eval: {len(eval_df)} righe")