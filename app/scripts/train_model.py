import tensorflow as tf
import json
import numpy as np
from pathlib import Path
from tensorflow.keras import layers as tfl
from app.src.models.model_computing import ModelComputing


def inspect_dataset(dataset, num_samples=1):
    """
    Inspecciona un tf.data.Dataset: tipos de datos, dimensiones, total de muestras y un sample.
    """
    print("\n" + "="*50)
    print("INSPECCIÓN DEL DATASET")
    print("="*50)
    
    # 0. Determinar tamaño del lote y cardinalidad
    cardinality = dataset.cardinality().numpy()
    
    # Intentamos obtener el batch_size del primer elemento
    first_batch = next(iter(dataset.take(1)))
    if isinstance(first_batch, tuple):
        # Si es (features, labels), miramos la primera feature o la label
        example_data = first_batch[1] # Usamos labels que suele ser un tensor simple
    else:
        example_data = first_batch
    
    batch_size = example_data.shape[0] if len(example_data.shape) > 0 else 1
    
    if cardinality == tf.data.INFINITE_CARDINALITY:
        print("\n[Tamaño del Dataset]: Infinito")
    elif cardinality == tf.data.UNKNOWN_CARDINALITY:
        print("\n[Tamaño del Dataset]: Desconocido")
    else:
        total_samples = cardinality * batch_size
        print(f"\n[Configuración]:")
        print(f"  - Lotes totales (m_batches): {cardinality}")
        print(f"  - Tamaño del lote (batch_size): {batch_size}")
        print(f"  - MUESTRAS TOTALES (m_total): {total_samples:,}")

    # 1. Mostrar estructura (element_spec)
    print("\n[Estructura del Dataset (element_spec)]:")
    spec = dataset.element_spec
    # ... resto de la función igual
    if isinstance(spec, tuple):
        print(f"Tipo: Tuple (Features, Labels/Targets)")
        print(f"Features: {spec[0]}")
        print(f"Labels:   {spec[1]}")
    else:
        print(f"Spec: {spec}")

    # 2. Mostrar un sample
    print(f"\n[Sample de {num_samples} lote(s)]:")
    for i, batch in enumerate(dataset.take(num_samples)):
        print(f"\n--- Batch {i+1} ---")
        if isinstance(batch, tuple):
            features, labels = batch
            print("FEATURES:")
            if isinstance(features, dict):
                for k, v in features.items():
                    print(f"  - {k:15}: Shape={v.shape}, Dtype={v.dtype}")
            else:
                print(f"  - Shape={features.shape}, Dtype={features.dtype}")
            
            print("\nLABELS/TARGETS:")
            print(f"  - Shape={labels.shape}, Dtype={labels.dtype}")
            
            # Mostrar una pequeña parte de los valores reales
            print("\nVALORES (Primeros 2 elementos del lote):")
            if isinstance(features, dict):
                for k, v in features.items():
                    print(f"  {k}: {v.numpy()[:2]}")
            else:
                print(f"  Features sample: {features.numpy()[:2]}")
            print(f"  Labels sample:   {labels.numpy()[:2]}")
        else:
            print(f"Data: {batch}")
    
    print("="*50 + "\n")


def main():
    # Rutas de archivos
    DATASET_PATH = Path("app/data/processed/tf_dataset")
    METADATA_PATH = Path("app/models/metadata/model_metadata.json")
    MODEL_SAVE_PATH = Path("app/models/checkpoints/global_model_v1.keras")

    # A. Cargar Metadatos
    print("Loading metadata...")
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    # B. Cargar Dataset
    print("Loading processed dataset...")
    full_ds = tf.data.Dataset.load(str(DATASET_PATH))
    
    # Inspeccionar el dataset
    inspect_dataset(full_ds)


    config_model = {
        "inputs_lags": 21,
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
        "outputs": ["log_return"],
        "base_model": "",
        "hyperparameters_init": {
            "lr" : 2e-3,
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

    model_computing = ModelComputing(ds_tf=full_ds, config=config_model)
    model_computing.generate_splits()

if __name__ == "__main__":
    main()