import tensorflow as tf
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

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

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
                lambda features, labels: tf.gather(features["input_numerical"], target_indices, axis=-1),
                num_parallel_calls=tf.data.AUTOTUNE
            )

            sample_size = 20000 
            target_ds_sampled = target_ds.shuffle(buffer_size=10000).take(sample_size)
            scaler_instance = scalers_executions[scaler_name](target_ds_sampled)
            scaler_config.instance = scaler_instance
    
    def prepare_datasets(self):
        pass

            

            
