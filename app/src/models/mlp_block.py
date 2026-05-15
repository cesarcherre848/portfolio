import tensorflow as tf
from tensorflow.keras import layers as tfl

class MLPBlock(tfl.Layer):
    """
    Bloque final de decisión (Multi-Layer Perceptron).
    Recibe la combinación de características y produce la predicción final.
    """
    def __init__(self, output_dim, hidden_units=[32, 16], dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        
        # Construimos la pila de capas densas
        layers = []
        for units in hidden_units:
            layers.append(tfl.Dense(units, activation="relu"))
            layers.append(tfl.Dropout(dropout))
            
        self.hidden_layers = tf.keras.Sequential(layers)
        
        # Capa de salida ajustada al horizonte de predicción (outputs_horizons)
        self.output_layer = tfl.Dense(output_dim, name="output")

    def call(self, inputs, training=None):
        
        x = self.hidden_layers(inputs, training=training)
        x = tfl.Dropout(0.2)(x, training=training)
        return self.output_layer(x)
