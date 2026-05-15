import tensorflow as tf
from tensorflow.keras import layers as tfl
from app.src.models.hidden_blocks.sequence_block import get_sequence_block

class NumericalBlock(tfl.Layer):

    def __init__(self, block_type: str = "lstm", block_params: dict = None, **kwargs):
        super().__init__(**kwargs)
        
   
        if block_params is None:
            block_params = {}

        self.sequence_processor = get_sequence_block(block_type, **block_params)

    def call(self, inputs, training=None):
        
        return self.sequence_processor(inputs, training=training)
