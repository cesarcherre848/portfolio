import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import json

# --- CONFIGURACIÓN DEL CONTRATO DE DATOS ---
# El orden aquí es SAGRADO para la Rama Temporal del modelo
TEMP_FEATURES = [
    'log_return', 
    'volume_prc', 
    'delta_fed_rate', 
    'yield_spread', 
    'vix_log_return'
]
STATIC_FEATURES = ['ticker', 'sector']
TARGET_COL = 'log_return'
N_LAGS = 21
BATCH_SIZE = 64

def save_metadata(df, folder_path):
    """Guarda vocabularios y orden de columnas para asegurar consistencia."""
    path = Path(folder_path)
    path.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "temp_features_order": TEMP_FEATURES,
        "n_lags": N_LAGS,
        "target_col": TARGET_COL,
        "ticker_vocab": sorted(df['ticker'].unique().tolist()),
        "sector_vocab": sorted(df['sector'].unique().tolist())
    }
    
    with open(path / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"✅ Metadatos guardados en {path}")

def create_windowed_dataset(df):
    """Transforma el DataFrame en cubos (lags) respetando los grupos por ticker."""
    x_temp_all = []
    x_ticker_all = []
    x_sector_all = []
    y_all = []

    print("Generando ventanas temporales (cubos)...")
    
    # Agrupamos para evitar que el final de un ticker se mezcle con el inicio de otro
    for ticker, group in df.groupby('ticker'):
        # Forzamos el orden de las columnas para la rama temporal
        temp_data = group[TEMP_FEATURES].values
        ticker_vals = group['ticker'].values
        sector_vals = group['sector'].values
        targets = group[TARGET_COL].values

        # Deslizamos la ventana
        for i in range(N_LAGS, len(group)):
            # Rama Temporal: Matriz [N_LAGS x 5]
            x_temp_all.append(temp_data[i-N_LAGS:i])
            
            # Rama Estática: Strings individuales
            x_ticker_all.append(ticker_vals[i])
            x_sector_all.append(sector_vals[i])
            
            # Target: Log-return del día siguiente (t+1)
            y_all.append(targets[i])

    # Convertir a Tensores de TensorFlow
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            "input_temporal": np.array(x_temp_all, dtype='float32'),
            "ticker": np.array(x_ticker_all),
            "sector": np.array(x_sector_all)
        },
        np.array(y_all, dtype='float32')
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

    # 3. Crear Dataset de Ventanas
    full_ds = create_windowed_dataset(df)

    # 4. Configurar para entrenamiento (Shuffle y Batch)
    train_ds = full_ds.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # 5. Persistir el Dataset de TF
    save_path = "app/data/processed/tf_dataset"
    tf.data.Dataset.save(train_ds, save_path)
    print(f"🚀 Dataset de TensorFlow guardado exitosamente en: {save_path}")

if __name__ == "__main__":
    main()