import tensorflow as tf
from typing import Dict, Any, Optional


class ModelComputing:
    def __init__(self, ds_tf: Optional[tf.data.Dataset] = None, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa la clase ModelComputing usando TensoFlow Datasets.
        
        :param dataset_tf: Un objeto tf.data.Dataset ya procesado.
        :param config: Diccionario con hiperparámetros (batch_size, epochs, learning_rate, etc.).
        """
        self.ds = ds_tf
        self.config = config if config is not None else {}

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
    
    def generate_splits(self):
        if self.ds is None:
            raise ValueError("No se ha proporcionado un dataset_tf.")

        # Ahora total_batches representa la cantidad de lotes, no de filas
        total_batches = self.ds.cardinality().numpy()
        
        if total_batches <= 0:
            # Si es infinito o desconocido, el split fallará
            raise ValueError("El dataset tiene una estructura de lotes desconocida.")

        split_cfg = self.config.get("split", {"train": 0.9, "val": 0.05, "test": 0.05})
        
        # Calculamos cuántos BATCHES van a cada set
        train_count = int(total_batches * split_cfg["train"])
        val_count = int(total_batches * split_cfg["val"])
        test_count = total_batches - (train_count + val_count)

        print(f"Dividiendo {total_batches} batches totales:")
        print(f" -> Train: {train_count} batches")
        print(f" -> Val: {val_count} batches")
        print(f" -> Test: {test_count} batches")

        self.train_ds = self.ds.take(train_count)
        
        remaining = self.ds.skip(train_count)
        self.val_ds = remaining.take(val_count)
        self.test_ds = remaining.skip(val_count)


        self.train_ds = self.train_ds.prefetch(tf.data.AUTOTUNE)
        self.val_ds = self.val_ds.prefetch(tf.data.AUTOTUNE)
        self.test_ds = self.test_ds.prefetch(tf.data.AUTOTUNE)

        return self.train_ds, self.val_ds, self.test_ds
    
