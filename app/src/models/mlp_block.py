import tensorflow as tf
from tensorflow.keras import layers as tfl

class MLPBlock(tfl.Layer):
    """
    Bloque final de decisión (Multi-Layer Perceptron).
    Recibe la combinación de características y produce la predicción final.
    """
    def __init__(self, output_dim, hidden_units=[32, 16], dropout=0.2, l2_reg=0.01, **kwargs):
        super().__init__(**kwargs)
        
        # Construimos la pila de capas densas
        layers = []
        for units in hidden_units:
            layers.append(tfl.Dense(
                units, 
                activation="relu", 
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            ))
            layers.append(tfl.Dropout(dropout))
            
        self.hidden_layers = tf.keras.Sequential(layers)
        
        # Capa de dropout final
        self.dropout_layer = tfl.Dropout(dropout)

        # Capa de salida ajustada al horizonte de predicción (outputs_horizons)
        self.output_layer = tfl.Dense(
            output_dim, 
            name="output",
            kernel_initializer="glorot_uniform"
        )

    def call(self, inputs, training=None):
        
        x = self.hidden_layers(inputs, training=training)
        x = self.dropout_layer(x, training=training)
        return self.output_layer(x)
