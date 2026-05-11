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
            "main_net": {
                "hidden_block": ""
            }
        }
    }

    config_model = ConfigModel.from_dict(config_model_dict)
    #print(config_model)

    model_computing = ModelComputing(ds_tf=full_ds, config=config_model)
    model_computing.generate_splits()
    model_computing.compute_scalers()
    model_computing.prepare_datasets()
    
    # Inspeccionar muestras de un ticker específico para validar consistencia y ver las fechas
    model_computing.inspect_ready_samples(ds_type="train", num_samples=2, ticker="AAPL")

if __name__ == "__main__":
    main()