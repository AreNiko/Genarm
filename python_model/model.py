import tensorflow as tf
from tensorflow.keras import layers, models

def 3DModel(input_shape):

	xdim, ydim, zdim, channels = input_shape

	structure = layers.Input(input_shape)
	lay = 1
	model = models.Model(inputs=structure, outputs=lay)

    return model