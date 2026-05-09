import tensorflow as tf
from tensorflow.keras import Model

class BaseModel(Model):
	def __init__(self, config_net):
		super().__init__()