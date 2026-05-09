import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import json

# --- CONFIGURACIÓN DEL CONTRATO DE DATOS ---
TEMP_FEATURES = [
    'log_return', 
    'volume_prc', 
    'delta_fed_rate', 
    'yield_spread', 
    'vix_log_return'
]
STATIC_FEATURES = ['ticker', 'sector']
TIME_COL = 'date' # Agregamos la referencia explícita a la columna de tiempo
TARGET_COL = 'log_return'

def save_metadata(df, folder_path):
    """Guarda vocabularios y orden de columnas para asegurar consistencia."""
    path = Path(folder_path)
    path.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "temp_features_order": TEMP_FEATURES,
        "target_col": TARGET_COL,
        "ticker_vocab": sorted(df['ticker'].unique().tolist()),
        "sector_vocab": sorted(df['sector'].unique().tolist())
    }
    
    with open(path / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"✅ Metadatos guardados en {path}")

def create_flat_dataset(df):
    """Transforma el DataFrame en un Dataset tabular puro (fila por fila)."""
    print("Mapeando columnas al formato tf.data.Dataset...")
    
    # 1. Aseguramos el orden estricto antes de extraer las matrices
    df = df.sort_values(by=['ticker', TIME_COL])
    
    # 2. Extraemos las matrices usando vectorización
    x_num = df[TEMP_FEATURES].values.astype('float32')
    x_ticker = df['ticker'].values
    x_sector = df['sector'].values
    
    # NUEVO: Extraemos la fecha y la convertimos a string para compatibilidad TF
    x_date = df[TIME_COL].astype(str).values 
    
    y = df[TARGET_COL].values.astype('float32')

    # 3. Convertimos a Tensores de TensorFlow, inyectando la fecha
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            "input_numerical": x_num,
            "ticker": x_ticker,
            "sector": x_sector,
            "date": x_date # Ahora la fecha viaja junto a cada muestra
        },
        y
    ))

    return dataset

def main():
    # 1. Cargar datos crudos
    file_path = Path("app/data/processed/data.parquet")
    if not file_path.exists():
        print(f"❌ Error: No se encontró {file_path}")
        return

    df = pd.read_parquet(file_path)
    df = df.sort_values(['ticker', 'date']) # Asegurar orden cronológico
    
    # 2. Persistir metadatos (Orden de columnas e índices)
    save_metadata(df, "app/models/metadata")

    # 3. Crear Dataset Plano (Sin lags)
    raw_ds = create_flat_dataset(df)

    # 4. Persistir el Dataset de TF (Sin batch ni shuffle)
    save_path = "app/data/processed/tf_dataset_raw"
    tf.data.Dataset.save(raw_ds, save_path)
    
    print(f"🚀 Dataset crudo de TensorFlow guardado exitosamente en: {save_path}")
    print(f"Total de registros mapeados: {raw_ds.cardinality().numpy()}")

if __name__ == "__main__":
    main()