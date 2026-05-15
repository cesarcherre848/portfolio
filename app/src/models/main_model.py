import tensorflow as tf
from tensorflow.keras import layers as tfl
from app.src.models.categorical_block import CategoricalBlock
from app.src.models.numerical_block import NumericalBlock
from app.src.models.mlp_block import MLPBlock

class MainModel(tf.keras.Model):
    """Modelo modular que orquesta las ramas categórica y numérica."""
    def __init__(self, config, ticker_vocab, sector_vocab, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # Mapeo de nombres de variables numéricas a sus índices para el escalado
        self.num_var_to_idx = {name: idx for idx, name in enumerate(config.inputs.numerical)}
        
        # Inicializamos los bloques modulares
        self.cat_block = CategoricalBlock(ticker_vocab, sector_vocab)
        
        # El NumericalBlock recibe la configuración del sequence_block desde el config
        self.num_block = NumericalBlock(
            block_type=config.sequence_block_type,
            block_params=config.sequence_block_params
        )
        
        self.fusion_block = tfl.Concatenate()
        
        # El bloque de salida MLP configurado dinámicamente
        self.mlp_block = MLPBlock(
            output_dim=config.outputs_horizons,
            **config.mlp_block_params
        )

    def apply_scalers(self, numerical_inputs):
        """
        Aplica los escaladores configurados a las columnas específicas del tensor numérico.
        numerical_inputs shape: (batch, lags, num_features)
        """
        # Descomponemos el tensor en columnas (axis=-1)
        columns = tf.unstack(numerical_inputs, axis=-1)
        scaled_columns = list(columns)
        
        for scaler_name, scaler_struct in self.config.scalers.items():
            if scaler_struct.instance is None:
                continue
                
            # Identificar índices de las columnas que este escalador debe procesar
            indices = [self.num_var_to_idx[col_name] for col_name in scaler_struct.inputs]
            
            # Extraer las columnas específicas para este escalador
            to_scale = tf.gather(numerical_inputs, indices, axis=-1)
            
            # Aplicar el escalador
            scaled_data = scaler_struct.instance(to_scale)
            
            # Reinsertar las columnas escaladas
            unstacked_scaled = tf.unstack(scaled_data, axis=-1)
            for i, col_idx in enumerate(indices):
                scaled_columns[col_idx] = unstacked_scaled[i]
                
        # Recomponer el tensor numérico
        return tf.stack(scaled_columns, axis=-1)

    def call(self, inputs, training=None):
        """
        Flujo del modelo:
        inputs: dict con llaves ["ticker", "sector", "numerical"]
        """
        # 0. Aplicar Escaladores a la entrada numérica
        x_num_scaled = self.apply_scalers(inputs["numerical"])
        
        # 1. Procesar ramas categórica y numérica
        x_cat = self.cat_block(inputs["ticker"], inputs["sector"])
        x_num = self.num_block(x_num_scaled, training=training)
        
        # 2. Fusión (Concatenación)
        combined = self.fusion_block([x_num, x_cat])
        
        # 3. Predicción final
        return self.mlp_block(combined, training=training)
