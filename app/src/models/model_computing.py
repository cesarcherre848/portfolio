import tensorflow as tf
import numpy as np
from typing import Dict, Any, Optional
from app.src.models.structs.config_model import ConfigModel
from app.src.models.utils.scalers import scalers_executions


class ModelComputing:
    def __init__(self, ds_tf: Optional[tf.data.Dataset] = None, config: ConfigModel = None):
        """
        Inicializa la clase ModelComputing usando TensoFlow Datasets.

        :param dataset_tf: Un objeto tf.data.Dataset ya procesado.
        :param config: Diccionario con hiperparámetros (batch_size, epochs, learning_rate, etc.).
        """
        self.ds = ds_tf
        self.config = config

        # Datasets crudos (post-split, pre-pipeline)
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        # Datasets listos para entrenamiento (post-pipeline)
        self.train_ds_ready = None
        self.val_ds_ready = None
        self.test_ds_ready = None

        self.compute_indexes()

    def generate_splits(self):
        if self.ds is None:
            raise ValueError("No se ha proporcionado un dataset_tf.")

        # Obtenemos la cardinalidad (cantidad total de registros si es raw, o de lotes si tiene batch)
        total_batches = self.ds.cardinality().numpy()

        if total_batches <= 0 or total_batches == tf.data.UNKNOWN_CARDINALITY:
            raise ValueError("El dataset tiene una estructura de lotes desconocida o infinita.")

        # ==========================================
        # LECTURA DESDE EL STRUCT (ConfigModel)
        # ==========================================
        train_ratio = self.config.split.train
        val_ratio = self.config.split.val

        # Calculamos cuántos elementos van a cada set
        train_count = int(total_batches * train_ratio)
        val_count = int(total_batches * val_ratio)
        test_count = total_batches - (train_count + val_count)

        print(f"Dividiendo {total_batches} elementos totales:")
        print(f" -> Train: {train_count}")
        print(f" -> Val:   {val_count}")
        print(f" -> Test:  {test_count}")

        # Ejecutamos las particiones secuenciales
        self.train_ds = self.ds.take(train_count)

        remaining = self.ds.skip(train_count)
        self.val_ds = remaining.take(val_count)
        self.test_ds = remaining.skip(val_count)

        # Optimizamos el pipeline
        self.train_ds = self.train_ds.prefetch(tf.data.AUTOTUNE)
        self.val_ds = self.val_ds.prefetch(tf.data.AUTOTUNE)
        self.test_ds = self.test_ds.prefetch(tf.data.AUTOTUNE)

        return self.train_ds, self.val_ds, self.test_ds

    def compute_indexes(self):

        if not self.config.inputs.numerical:
            raise ValueError("❌ No hay variables numéricas definidas en la configuración (inputs.numerical).")

        master_num_features = self.config.inputs.numerical

        self.name_to_idx = {name: idx for idx, name in enumerate(master_num_features)}
        self.idx_to_name = {idx: name for idx, name in enumerate(master_num_features)}



    def compute_scalers(self):

        if not self.train_ds:
            raise ValueError("❌ El dataset de entrenamiento (train_ds) no existe. Ejecuta generate_splits() primero.")

        current_scalers = list(self.config.scalers.keys())
        available_scalers = list(scalers_executions.keys())

        missing_scalers = set(current_scalers) - set(available_scalers)

        if missing_scalers:
            raise ValueError(
                f"Los siguientes scalers configurados no están disponibles en scalers_executions: {missing_scalers}. "
                f"Asegúrate de que estén definidos en app.src.models.utils.scalers."
            )


        for scaler_name in current_scalers:
            scaler_config = self.config.scalers[scaler_name]
            target_cols = scaler_config.inputs

            target_indices = [self.name_to_idx[col] for col in target_cols]

            target_ds = self.train_ds.map(
                lambda features: tf.gather(features["numerical"], target_indices, axis=-1),
                num_parallel_calls=tf.data.AUTOTUNE)

            sample_size = 20000 
            target_ds_sampled = target_ds.shuffle(buffer_size=10000).take(sample_size)
            scaler_instance = scalers_executions[scaler_name](target_ds_sampled)
            scaler_config.instance = scaler_instance

    def build_pipeline(self, ds: tf.data.Dataset):
        """
        Construye el pipeline de datos: agrupación por ticker, lagging (lags + padding NaN),
        generación de X, Y y creación de batches.
        """
        lags = self.config.inputs_lags
        batch_size = self.config.hyperparameters_init.batch_size
        num_features = len(self.config.inputs.numerical)

        def create_windows(key, group_ds):
            # 1. Padding de NaNs al inicio de cada ticker
            # Solo añadimos 'lags' filas de NaNs para permitir que el primer registro tenga ventana completa.
            nan_record = {
                "numerical": tf.fill([num_features], tf.constant(float('nan'), dtype=tf.float32)),
                "ticker": tf.constant(""),
                "sector": tf.constant(""),
                "date": tf.constant("")
            }
            padding_ds = tf.data.Dataset.from_tensors(nan_record).repeat(lags)

            # Concatenamos padding + datos reales
            full_group_ds = padding_ds.concatenate(group_ds)

            # 2. Ventaneo: lags + 1 (el +1 es para el target Y)
            windowed_ds = full_group_ds.window(size=lags + 1, shift=1, drop_remainder=True)

            # 3. Convertir cada ventana (que es un dict de datasets) en un batch (tensor)
            # Usamos Dataset.zip(w) porque w es un diccionario de datasets
            return windowed_ds.flat_map(lambda w: tf.data.Dataset.zip(w).batch(lags + 1))

        # Agrupamos por ticker. key_func requiere int64, usamos un hash.
        # Al estar los datos ya ordenados por ticker, group_by_window procesará uno por uno eficientemente.
        ds = ds.group_by_window(
            key_func=lambda x: tf.strings.to_hash_bucket_fast(x['ticker'], num_buckets=20000),
            reduce_func=create_windows,
            window_size=tf.constant(1000000, dtype=tf.int64) 
        )

        def split_xy(window):
            # X: los primeros 'lags' elementos
            # Y: el valor de 'log_return' del último elemento (índice 'lags')
            log_return_idx = self.name_to_idx.get("log_return", 0)

            x_numerical = window["numerical"][:lags]
            y = window["numerical"][lags, log_return_idx]

            features = {
                "numerical": x_numerical,
                "ticker": window["ticker"][-1], # Tomamos el ticker del registro objetivo (Y)
                "sector": window["sector"][-1],
                "date": window["date"][-1]      # Preservamos la fecha del objetivo
            }

            return features, y

        # Mapeamos a X, Y
        ds = ds.map(split_xy, num_parallel_calls=tf.data.AUTOTUNE)

        # Generamos los batches finales
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return ds

    def prepare_datasets(self):
        """
        Aplica el build_pipeline a los sets de train, val y test,
        almacenando el resultado en los atributos '_ready'.
        """
        if self.train_ds is None:
            self.generate_splits()

        print("Preparando pipelines de datos (windowing, lagging, batching)...")
        self.train_ds_ready = self.build_pipeline(self.train_ds)
        self.val_ds_ready = self.build_pipeline(self.val_ds)
        self.test_ds_ready = self.build_pipeline(self.test_ds)
        print("✅ Pipelines listos y guardados en atributos '_ready'.")

    def inspect_ready_samples(self, ds_type: str = "train", num_samples: int = 1, ticker: Optional[str] = None):
        """
        Muestra una vista detallada de las muestras en el dataset 'ready' para validación.
        Permite filtrar por un ticker específico.
        """
        ds_map = {
            "train": self.train_ds_ready,
            "val": self.val_ds_ready,
            "test": self.test_ds_ready
        }
        
        ds = ds_map.get(ds_type)
        if ds is None:
            print(f"❌ El dataset '{ds_type}_ready' no está inicializado.")
            return

        print(f"\n--- Inspeccionando {num_samples} muestras de {ds_type}_ready ---")
        if ticker:
            print(f"Filtrando por Ticker: {ticker}")
            # Filtramos (requiere des-batchar temporalmente para aplicar el filtro por elemento)
            ds = ds.unbatch().filter(lambda f, y: tf.equal(f['ticker'], ticker)).batch(self.config.hyperparameters_init.batch_size)

        # Tomamos un batch y extraemos muestras
        found = 0
        for features, target in ds.take(5): # Buscamos en hasta 5 batches si hay filtro
            if found >= num_samples: break
            
            batch_len = features['ticker'].shape[0]
            for i in range(batch_len):
                if found >= num_samples: break
                
                t_curr = features['ticker'][i].numpy().decode('utf-8')
                d_curr = features['date'][i].numpy().decode('utf-8')
                s_curr = features['sector'][i].numpy().decode('utf-8')
                
                print(f"\nSample {found+1}:")
                print(f"  Ticker: {t_curr}")
                print(f"  Date:   {d_curr}")
                print(f"  Sector: {s_curr}")
                print(f"  X (numerical) shape: {features['numerical'][i].shape}")
                
                # Secuencia completa de lags
                lags_full = features['numerical'][i].numpy()
                print(f"  X (secuencia completa de {lags_full.shape[0]} lags):")
                for idx, lag in enumerate(lags_full):
                    print(f"    Lag {idx:2d}: {lag}")
                print(f"  Y (target log_return): {target[i].numpy():.6f}")
                
                if np.isnan(target[i].numpy()):
                    print("  ⚠️ Nota: El target es NaN (inicio de serie)")
                
                found += 1
        
        if found == 0:
            print(f"⚠️ No se encontraron muestras" + (f" para el ticker '{ticker}'" if ticker else ""))






            

            
