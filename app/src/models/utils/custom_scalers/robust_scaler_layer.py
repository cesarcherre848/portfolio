import tensorflow as tf
import tensorflow_probability as tfp

@tf.keras.utils.register_keras_serializable()
class RobustScalerLayer(tf.keras.layers.Layer):

    def __init__(self, column_names=None, **kwargs):
        super().__init__(**kwargs)
        self.median = None
        self.iqr = None

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.median = self.add_weight(
            name="median",
            shape=(input_dim,),
            initializer="zeros",
            trainable=False
        )
        self.iqr = self.add_weight(
            name="iqr",
            shape=(input_dim,),
            initializer="ones",
            trainable=False
        )
        super().build(input_shape)

    def adapt(self, data):
        if isinstance(data, tf.data.Dataset):
            # Consumimos el generador lazy
            tensors = list(data)
            
            # Concatenamos según la estructura (batcheado vs unbatched)
            if len(tensors[0].shape) == 1:
                data = tf.stack(tensors, axis=0)
            else:
                data = tf.concat(tensors, axis=0)
        
        if not self.built:
            self.build(data.shape)

        data = tf.cast(data, tf.float32)
        median = tfp.stats.percentile(data, 50.0, axis=0, interpolation='midpoint')
        q1 = tfp.stats.percentile(data, 25.0, axis=0, interpolation='midpoint')
        q3 = tfp.stats.percentile(data, 75.0, axis=0, interpolation='midpoint')
        iqr = tf.where(q3 - q1 == 0.0, tf.ones_like(q3), q3 - q1)
        self.median.assign(median)
        self.iqr.assign(iqr)


    @tf.function
    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        return tf.divide(tf.subtract(inputs, self.median), self.iqr)

    @tf.function
    def inverse(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        return tf.add(tf.multiply(inputs, self.iqr), self.median)


@tf.keras.utils.register_keras_serializable()
class InverseRobustScalerLayer(tf.keras.layers.Layer):
    def __init__(self, scaler_layer, **kwargs):
        super().__init__(**kwargs)
        self.scaler_layer = scaler_layer 

    @tf.function
    def call(self, inputs):
        return self.scaler_layer.inverse(inputs)