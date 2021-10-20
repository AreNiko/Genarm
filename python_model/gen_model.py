import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def Model3D(input_shape):

	xdim, ydim, zdim = input_shape
	print(input_shape)
	struct = layers.Input(input_shape)
	x = layers.Dense(32)(struct)
	x = layers.Dense(64)(x)
	x = layers.Dense(128)(x)
	x = layers.Dense(256)(x)
	x = layers.LeakyReLU(alpha=0.2)(x)
	#x = layers.BatchNormalization(momentum=0.8)(x)
	x = layers.Dense(512)(x)
	x = layers.LeakyReLU(alpha=0.2)(x)
	#x = layers.BatchNormalization(momentum=0.8)(x)
	x = layers.Dense(1024)(x)
	x = layers.LeakyReLU(alpha=0.2)(x)
	#x = layers.BatchNormalization(momentum=0.8)(x)
	x = layers.Dense(512)(x)
	x = layers.LeakyReLU(alpha=0.2)(x)
	#x = layers.BatchNormalization(momentum=0.8)(x)
	x = layers.Dense(128)(x)
	x = layers.LeakyReLU(alpha=0.2)(x)
	#x = layers.BatchNormalization(momentum=0.8)(x)
	x = layers.Dense(64)(x)
	x = layers.LeakyReLU(alpha=0.2)(x)
	#x = layers.BatchNormalization(momentum=0.8)(x)
	x = layers.Dense(zdim)(x)
	#x = layers.Flatten()(x) 
	"""
	x = layers.Reshape((xdim, ydim, zdim,1))(struct) 
	x = layers.Conv3D(16, 5, strides=(1, 1, 1), padding='same', activation='relu')(x)
	x = layers.MaxPooling3D((2, 2, 2), strides=1, padding='same')(x)
	x = layers.Conv3D(32, 5, strides=(1, 1, 1), padding='same', activation='relu')(x)
	x = layers.Conv3DTranspose(1, (2,2,2), strides=1, padding='same')(x)
	#x = layers.Conv3D(zdim, 1, padding='same', activation=None)(x)
	"""
	x = tf.keras.activations.tanh(x)
	#x = tf.math.ceil(x)
	#x = layers.Flatten()(x) 
	#x = layers.Reshape((5,5,5))(x) 
	#x = layers.Dense(zdim)(x)
	

	
	#x = tf.cast(x, tf.float32)

	model = models.Model(inputs=struct, outputs=x)
	"""
	model = models.Sequential()

	model.add(layers.Input(input_shape))
	#model.add(layers.Flatten())
	
	#model.add(layers.Flatten(input_shape=input_shape))
	#model.add(layers.Dense(8))
	#model.add(layers.Dense(16))
	model.add(layers.Dense(32))
	model.add(layers.Dense(64))
	model.add(layers.Dense(128))
	#model.add(layers.Dense(256))
	#model.add(layers.Dense(512))
	model.add(layers.Dense(np.prod(input_shape), activation='tanh'))
	model.add(layers.Reshape(input_shape))
	"""
	#return model
	#structure = layers.Input(input_shape)
	#new_struct = model(structure)
	
	return model

def ConvModel3D(input_shape):

	xdim, ydim, zdim = input_shape
	input_shape = (xdim, ydim, zdim, 1)
	print(input_shape)
	struct = layers.Input(input_shape)
	"""
	x = layers.Conv2D(32, 5, strides=1, padding='same', activation='relu')(struct)
	x = layers.Conv2D(64, 5, strides=1, padding='same', activation='relu')(x)
	#x = layers.Conv2D(128, 5, strides=1, padding='same', activation='relu')(x)
	#x = layers.Conv2D(256, 5, strides=1, padding='same', activation='relu')(x)
	#x = layers.Conv2D(512, 5, strides=1, padding='same', activation='relu')(x)

	#x = layers.Conv2D(256, 5, strides=1, padding='same', activation='relu')(x)
	#x = layers.Conv2D(128, 5, strides=1, padding='same', activation='relu')(x)
	x = layers.Conv2D(64, 5, strides=1, padding='same', activation='relu')(x)
	x = layers.Conv2D(32, 5, strides=1, padding='same', activation='relu')(x)
	x = layers.Conv2D(zdim, 1, padding='same', activation=None)(x)
	#x = layers.BatchNormalization(momentum=0.8)(x
	#x = layers.Flatten()(x) 
	"""

	x = layers.Conv3D(32, 5, strides=(1, 1, 1), padding='same', activation='relu')(struct)
	x = layers.Conv3D(64, 5, strides=(1, 1, 1), padding='same', activation='relu')(x)
	x = layers.Conv3D(128, 5, strides=(1, 1, 1), padding='same', activation='relu')(x)
	x = layers.Conv3D(256, 5, strides=(1, 1, 1), padding='same', activation='relu')(x)
	x = layers.Conv3D(256, 5, strides=(1, 1, 1), padding='same', activation='relu')(x)
	x = layers.Conv3D(128, 5, strides=(1, 1, 1), padding='same', activation='relu')(x)
	x = layers.Conv3D(64, 5, strides=(1, 1, 1), padding='same', activation='relu')(x)
	x = layers.Conv3D(32, 5, strides=(1, 1, 1), padding='same', activation='relu')(x)
	x = layers.Conv3D(1, 1, padding='same', activation='relu')(x)
	#x = layers.Conv3D(zdim, 1, padding='same', activation=None)(x)
	
	x = tf.keras.activations.tanh(x)
	#x = tf.math.ceil(x)
	#x = layers.Flatten()(x) 
	x = layers.Reshape((xdim, ydim, zdim))(x) 
	#x = layers.Dense(zdim)(x)
	

	
	#x = tf.cast(x, tf.float32)

	model = models.Model(inputs=struct, outputs=x)
	"""
	model = models.Sequential()

	model.add(layers.Input(input_shape))
	#model.add(layers.Flatten())
	
	#model.add(layers.Flatten(input_shape=input_shape))
	#model.add(layers.Dense(8))
	#model.add(layers.Dense(16))
	model.add(layers.Dense(32))
	model.add(layers.Dense(64))
	model.add(layers.Dense(128))
	#model.add(layers.Dense(256))
	#model.add(layers.Dense(512))
	model.add(layers.Dense(np.prod(input_shape), activation='tanh'))
	model.add(layers.Reshape(input_shape))
	"""
	#return model
	#structure = layers.Input(input_shape)
	#new_struct = model(structure)
	
	return model