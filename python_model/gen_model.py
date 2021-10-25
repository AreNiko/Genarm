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
	"""
	x = layers.Reshape((xdim, ydim, zdim,1))(struct) 
	x = layers.Conv3D(16, 5, strides=(1, 1, 1), padding='same', activation='relu')(x)
	x = layers.MaxPooling3D((2, 2, 2), strides=1, padding='same')(x)
	x = layers.Conv3D(32, 5, strides=(1, 1, 1), padding='same', activation='relu')(x)
	x = layers.Conv3DTranspose(1, (2,2,2), strides=1, padding='same')(x)
	#x = layers.Conv3D(zdim, 1, padding='same', activation=None)(x)
	"""
	x = tf.keras.activations.tanh(x)

	model = models.Model(inputs=struct, outputs=x)
	
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
	#x = layers.BatchNormalization(momentum=0.8)(x)
	#x = layers.Flatten()(x) 
	"""

	x = layers.Conv3D(32, 5, strides=(1, 1, 1), padding='same', activation='relu')(struct)
	x = layers.BatchNormalization(momentum=0.8)(x)
	#x = layers.Conv3D(64, 5, strides=(1, 1, 1), padding='same', activation='relu')(x)
	#x = layers.BatchNormalization(momentum=0.8)(x)
	#x = layers.Conv3D(128, 5, strides=(1, 1, 1), padding='same', activation='relu')(x)
	#x = layers.BatchNormalization(momentum=0.8)(x)
	#x = layers.Conv3D(256, 5, strides=(1, 1, 1), padding='same', activation='relu')(x)
	#x = layers.BatchNormalization(momentum=0.8)(x)
	#x = layers.Conv3D(256, 5, strides=(1, 1, 1), padding='same', activation='relu')(x)
	#x = layers.BatchNormalization(momentum=0.8)(x)
	#x = layers.Conv3D(128, 5, strides=(1, 1, 1), padding='same', activation='relu')(x)
	#x = layers.BatchNormalization(momentum=0.8)(x)
	#x = layers.Conv3D(64, 5, strides=(1, 1, 1), padding='same', activation='relu')(x)
	#x = layers.BatchNormalization(momentum=0.8)(x)
	x = layers.Conv3D(32, 5, strides=(1, 1, 1), padding='same', activation='relu')(x)
	x = layers.BatchNormalization(momentum=0.8)(x)
	x = layers.Conv3D(1, 1, padding='same', activation='relu')(x)
	
	x = tf.keras.activations.tanh(x)
	x = layers.Reshape((xdim, ydim, zdim))(x) 

	model = models.Model(inputs=struct, outputs=x)
	
	return model

def ConvStructModel3D(input_shape):
	xdim, ydim, zdim = input_shape
	input_shape = (xdim, ydim, zdim, 1)
	print(input_shape)
	struct = layers.Input(input_shape)

	x1 = layers.Conv3D(128, 5, strides=(1, 1, 1), padding='same', activation='relu')(struct)
	x1 = layers.BatchNormalization(momentum=0.8)(x1)
	x1d = layers.Dense(128)(x1)

	x2 = layers.Conv3D(64, 5, strides=(1, 1, 1), padding='same', activation='relu')(x1)
	x2 = layers.BatchNormalization(momentum=0.8)(x2)
	x2d = layers.Dense(128)(x2)

	x3 = layers.Conv3D(32, 5, strides=(1, 1, 1), padding='same', activation='relu')(x2)
	x3 = layers.BatchNormalization(momentum=0.8)(x3)
	x3d = layers.Dense(128)(x3)

	x4 = layers.Conv3D(16, 5, strides=(1, 1, 1), padding='same', activation='relu')(x3)
	x4 = layers.BatchNormalization(momentum=0.8)(x4)
	x4d = layers.Dense(128)(x4)

	xf = tf.concat([x1d, x2d, x3d, x4d], -1)
	xfd = layers.Conv3D(1, 1, strides=(1, 1, 1), padding='same', activation='relu')(xf)
	
	xfd = tf.keras.activations.tanh(xfd)
	xfd = layers.Reshape((xdim, ydim, zdim))(xfd) 
	model = models.Model(inputs=struct, outputs=xfd)

	return model