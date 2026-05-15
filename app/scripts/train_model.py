import tensorflow as tf
import json
import numpy as np
from pathlib import Path
from tensorflow.keras import layers as tfl
from app.src.models.model_computing import ModelComputing
from app.src.models.structs.config_model import ConfigModel


def main():
    # Rutas de archivos
    DATASET_PATH = Path("app/data/processed/tf_dataset_raw")
    METADATA_PATH = Path("app/models/metadata/model_metadata.json")
    
    # A. Cargar Metadatos
    print("Loading metadata...")
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    # B. Cargar Dataset
    print("Loading processed dataset...")
    full_ds = tf.data.Dataset.load(str(DATASET_PATH))
    
    # Inspeccionar el dataset
    print(full_ds)

    config_model_dict = {
        "inputs_lags": 5,
        "outputs_horizons": 1,
        "inputs" : {
            "categorical": [
                "ticker",
                "sector"
            ],
            "numerical": [
                "log_return",
                "volume_prc",
                "delta_fed_rate",
                "yield_spread",
                "vix_log_return"
            ]
        },
        "scalers": {
            "RobustScalerLayer": {
                "inputs" : [
                    "log_return",
                    "volume_prc",
                    "delta_fed_rate",
                    "yield_spread",
                    "vix_log_return"
                ],
            }
        },

        "outputs": ["log_return"],
        "base_model": "",
        "hyperparameters_init": {
            "lr" : 2e-3,
            "batch_size": 64
        },
        "split": {
            "train": 0.9,
            "val": 0.05,
            "test" : 0.05
        },
        "models": {
            "main_model": {
                "sequence_block_type": "conv_stacked_lstm",
                "sequence_block_params": {
                    "filters": 64,
                    "units": [128, 64],
                    "dropout": 0.3
                },
                "mlp_block_params": {
                    "hidden_units": [64, 32],
                    "dropout": 0.2
                },
                "categorical_dims": {
                    "ticker": 16,
                    "sector": 4
                }
            }
        }
    }

    config_model = ConfigModel.from_dict(config_model_dict)

    model_computing = ModelComputing(ds_tf=full_ds, config=config_model)
    model_computing.generate_splits()
    model_computing.compute_scalers()
    model_computing.prepare_datasets()
    
    # --- VERIFICACIÓN DEL MODELO ---
    print("\n--- Verificando MainModel ---")
    
    # 1. Instanciar el modelo a través de ModelComputing
    model = model_computing.build_model(
        ticker_vocab=metadata["ticker_vocab"],
        sector_vocab=metadata["sector_vocab"]
    )

    # 2. Obtener un batch de ejemplo para construir el grafo y verificar formas
    sample_batch = next(iter(model_computing.train_ds_ready))
    features, targets = sample_batch

    # 3. Forward pass de prueba
    print(f"Feeding batch with shapes:")
    for k, v in features.items():
        print(f"  {k}: {v.shape}")
    print(f"  Targets: {targets.shape}")

    # En el MainModel.call ahora se aplican los scalers internamente
    outputs = model(features, training=False)
    
    print(f"\nModel Output Shape: {outputs.shape}")
    
    # 4. Resumen del modelo
    model.summary()

    print("\n✅ Verificación completada con éxito. Las dimensiones de los embeddings ahora son configurables.")

if __name__ == "__main__":
    main()
