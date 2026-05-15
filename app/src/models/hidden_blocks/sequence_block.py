import tensorflow as tf
from tensorflow.keras import layers as tfl

class BasicLSTMBlock(tfl.Layer):
    """Arquitectura: LSTM -> Dropout"""
    def __init__(self, units=64, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.lstm = tfl.LSTM(units)
        self.dropout = tfl.Dropout(dropout)

    def call(self, inputs, training=None):
        x = self.lstm(inputs)
        return self.dropout(x, training=training)

class ConvLSTMBlock(tfl.Layer):
    """Arquitectura: Conv1D -> LSTM -> Dropout"""
    def __init__(self, filters=32, kernel_size=3, units=64, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.conv = tfl.Conv1D(filters, kernel_size, activation='relu', padding='same')
        self.lstm = tfl.LSTM(units)
        self.dropout = tfl.Dropout(dropout)

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.lstm(x)
        return self.dropout(x, training=training)

class ConvStackedLSTMBlock(tfl.Layer):
    """Arquitectura: Conv1D -> LSTM -> LSTM -> Dropout"""
    def __init__(self, filters=32, kernel_size=3, units=[64, 32], dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.conv = tfl.Conv1D(filters, kernel_size, activation='relu', padding='same')
        self.lstm1 = tfl.LSTM(units[0], return_sequences=True)
        self.lstm2 = tfl.LSTM(units[1])
        self.dropout = tfl.Dropout(dropout)

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.lstm1(x)
        x = self.lstm2(x)
        return self.dropout(x, training=training)

class DeepHybridBlock(tfl.Layer):
    """Arquitectura: Conv1D -> Conv1D -> LSTM -> LSTM -> Dropout"""
    def __init__(self, filters=[32, 64], kernel_size=3, units=[64, 32], dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = tfl.Conv1D(filters[0], kernel_size, activation='relu', padding='same')
        self.conv2 = tfl.Conv1D(filters[1], kernel_size, activation='relu', padding='same')
        self.lstm1 = tfl.LSTM(units[0], return_sequences=True)
        self.lstm2 = tfl.LSTM(units[1])
        self.dropout = tfl.Dropout(dropout)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.lstm1(x)
        x = self.lstm2(x)
        return self.dropout(x, training=training)

def get_sequence_block(block_type: str, **kwargs):
    """Fábrica para obtener el bloque de secuencia por nombre."""
    blocks = {
        "basic_lstm": BasicLSTMBlock,
        "conv_lstm": ConvLSTMBlock,
        "conv_stacked_lstm": ConvStackedLSTMBlock,
        "deep_hybrid": DeepHybridBlock
    }
    
    if block_type not in blocks:
        raise ValueError(f"Block type '{block_type}' not recognized. Available: {list(blocks.keys())}")
        
    return blocks[block_type](**kwargs)
