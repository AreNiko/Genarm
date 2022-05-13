import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def Model3D(input_shape):

	xdim, ydim, zdim, channels = input_shape
	print(input_shape)
	struct = layers.Input(input_shape)

	x = layers.Conv3D(128, 5, strides=(1, 1, 1), padding='same', activation='relu')(struct)
	x = layers.Conv3D(64, 5, strides=(1, 1, 1), padding='same', activation='relu')(x)
	x = layers.Conv3D(32, 5, strides=(1, 1, 1), padding='same', activation='relu')(x)
	x = layers.Conv3D(16, 5, strides=(1, 1, 1), padding='same', activation='relu')(x)
	x = layers.Conv3D(1, 1, strides=(1, 1, 1), padding='same', activation='relu')(x)
	"""
	x = layers.Dense(32)(struct)
	x = layers.Dense(64)(x)
	x = layers.Dense(128)(x)
	x = layers.Dense(256)(x)
	x = layers.LeakyReLU(alpha=0.2)(x)
	#x = layers.BatchNormalization(momentum=0.8)(x)
	#x = layers.Dense(512)(x)
	#x = layers.LeakyReLU(alpha=0.2)(x)
	#x = layers.BatchNormalization(momentum=0.8)(x)
	#x = layers.Dense(1024)(x)
	#x = layers.LeakyReLU(alpha=0.2)(x)
	#x = layers.BatchNormalization(momentum=0.8)(x)
	#x = layers.Dense(512)(x)
	#x = layers.LeakyReLU(alpha=0.2)(x)
	#x = layers.BatchNormalization(momentum=0.8)(x)
	x = layers.Dense(128)(x)
	x = layers.LeakyReLU(alpha=0.2)(x)
	#x = layers.BatchNormalization(momentum=0.8)(x)
	x = layers.Dense(64)(x)
	x = layers.LeakyReLU(alpha=0.2)(x)

	x = layers.Dense(32)(x)
	x = layers.LeakyReLU(alpha=0.2)(x)
	#x = layers.BatchNormalization(momentum=0.8)(x)
	x = layers.Dense(1)(x)

	x = layers.Reshape((xdim, ydim, zdim,1))(struct)
	x = layers.Conv3D(16, 5, strides=(1, 1, 1), padding='same', activation='relu')(x)
	x = layers.MaxPooling3D((2, 2, 2), strides=1, padding='same')(x)
	x = layers.Conv3D(32, 5, strides=(1, 1, 1), padding='same', activation='relu')(x)
	x = layers.Conv3DTranspose(1, (2,2,2), strides=1, padding='same')(x)
	#x = layers.Conv3D(zdim, 1, padding='same', activation=None)(x)
	"""
	x = tf.keras.activations.sigmoid(x)

	x = layers.Reshape((xdim, ydim, zdim))(x)

	model = models.Model(inputs=struct, outputs=x)

	return model

def ConvModel3D(input_shape, structC, structF, vGstayOff):

	xdim, ydim, zdim, channels = input_shape
	input_shape = (xdim, ydim, zdim, channels)
	print(input_shape)


	xc, yc, zc, cc = structC
	xf, yf, zf, cf = structF
	outx, outy, outz, outc = vGstayOff
	c_shape = (xc, yc, zc, cc)
	f_shape = (xf, yf, zf, cf)
	stayoff_shape = (outx, outy, outz, outc)

	struct = layers.Input(input_shape)
	locked = layers.Input(c_shape)
	forces = layers.Input(f_shape)
	stayoff = layers.Input(stayoff_shape)

	x1 = layers.Conv3D(128, 5, strides=(1, 1, 1), padding='same', activation='relu')(struct)
	#x1 = layers.Conv3D(64, 5, strides=(1, 1, 1), padding='same')(struct)			# Without ReLu activation function
	#x1 = tf.keras.layers.MaxPool3D((2, 2, 2), padding='same')(x1)
	#x1 = layers.BatchNormalization(momentum=0.8)(x1)

	x2 = layers.Conv3D(64, 5, strides=(1, 1, 1), padding='same', activation='relu')(x1)
	#x2 = layers.Conv3D(64, 5, strides=(1, 1, 1), padding='same')(x1)			# Without ReLu activation function
	#x2 = layers.Conv3D(1, 1, strides=(1, 1, 1), padding='same', activation='relu')(x2)
	#x2 = layers.Conv3D(1, 1, strides=(1, 1, 1), padding='same')(x2)			# Without ReLu activation function
	#x2 = tf.keras.layers.MaxPool3D((2, 2, 2), padding='same')(x2)
	#x2 = layers.BatchNormalization(momentum=0.8)(x2)

	x3 = layers.Conv3D(32, 5, strides=(1, 1, 1), padding='same', activation='relu')(x2)
	#x3 = layers.Conv3D(64, 5, strides=(1, 1, 1), padding='same')(x2)			# Without ReLu activation function
	#x3 = layers.Conv3D(1, 1, strides=(1, 1, 1), padding='same', activation='relu')(x3)
	#x3 = layers.Conv3D(1, 1, strides=(1, 1, 1), padding='same')(x3)			# Without ReLu activation function
	#x3 = tf.keras.layers.MaxPool3D((2, 2, 2), padding='same')(x3)
	#x3 = layers.BatchNormalization(momentum=0.8)(x3)

	x4 = layers.Conv3D(16, 5, strides=(1, 1, 1), padding='same', activation='relu')(x3)
	#x4 = layers.Conv3D(64, 5, strides=(1, 1, 1), padding='same')(x3)			# Without ReLu activation function
	#x4 = layers.Conv3D(1, 1, strides=(1, 1, 1), padding='same', activation='relu')(x4)
	#x4 = layers.Conv3D(1, 1, strides=(1, 1, 1), padding='same')(x4)			# Without ReLu activation function
	#x4 = tf.keras.layers.MaxPool3D((2, 2, 2), padding='same')(x4)
	#x4 = layers.BatchNormalization(momentum=0.8)(x4)

	locked_1 = layers.Dense(128)(locked)
	forces_1 = layers.Dense(128)(forces)
	stayoff_1 = layers.Dense(128)(stayoff)
	x1 = tf.concat([x1, locked_1, forces_1, stayoff_1], -1)
	x1d = layers.Dense(32)(x1)
	#x1d = layers.Dense(32)(x1d)

	locked_2 = layers.Dense(64)(locked)
	forces_2 = layers.Dense(64)(forces)
	stayoff_2 = layers.Dense(64)(stayoff)
	x2 = tf.concat([x2, locked_2, forces_2, stayoff_2], -1)
	x2d = layers.Dense(32)(x2)
	#x2d = layers.Dense(32)(x2d)

	locked_3 = layers.Dense(32)(locked)
	forces_3 = layers.Dense(32)(forces)
	stayoff_3 = layers.Dense(32)(stayoff)
	x3 = tf.concat([x3, locked_3, forces_3, stayoff_3], -1)
	x3d = layers.Dense(32)(x3)
	#x3d = layers.Dense(32)(x3d)

	locked_4 = layers.Dense(16)(locked)
	forces_4 = layers.Dense(16)(forces)
	stayoff_4 = layers.Dense(16)(stayoff)
	x4 = tf.concat([x4, locked_4, forces_4, stayoff_4], -1)
	x4d = layers.Dense(32)(x4)
	#x4d = layers.Dense(32)(x4d)

	xf = tf.concat([x1d, x2d, x3d, x4d], -1)
	xf = layers.Dense(32)(xf)
	xfd = layers.Conv3D(1, 1, strides=(1, 1, 1), padding='same', activation='relu')(xf)

	xfd = tf.keras.activations.tanh(xfd)
	#xfd = tf.keras.activations.sigmoid(xfd)

	xfd = layers.Reshape((xdim, ydim, zdim))(xfd)
	#m = tf.constant(0.5)
	#xfd = tf.math.divide(tf.math.add(tf.math.subtract(xfd, m),tf.math.abs(tf.math.subtract(xfd, m))), tf.math.multiply(2.0,tf.math.abs(tf.math.subtract(xfd, m))))

	model = models.Model(inputs=[struct, locked, forces, stayoff], outputs=xfd)

	return model

def ConvStructModel3D(input_shape):
	xdim, ydim, zdim, channels = input_shape
	#input_shape = (xdim, ydim, zdim, 1)
	print(input_shape)
	struct = layers.Input(input_shape)

	x = layers.Conv3D(128, 5, strides=(1, 1, 1), padding='same', activation='relu')(struct)
	#x = layers.BatchNormalization(momentum=0.8)(x)

	x1 = layers.Conv3D(64, 5, strides=(1, 1, 1), padding='same', activation='relu')(x)
	#x1 = layers.BatchNormalization(momentum=0.8)(x1)

	x2 = layers.Conv3D(32, 5, strides=(1, 1, 1), padding='same', activation='relu')(x1)
	#x2 = layers.BatchNormalization(momentum=0.8)(x2)

	x3 = layers.Conv3D(16, 5, strides=(1, 1, 1), padding='same', activation='relu')(x2)
	#x3 = layers.BatchNormalization(momentum=0.8)(x3)

	x4 = layers.Conv3D(8, 5, strides=(1, 1, 1), padding='same', activation='relu')(x3)
	#x4 = layers.BatchNormalization(momentum=0.8)(x4)

	#x5 = layers.Conv3D(32, 5, strides=(1, 1, 1), padding='same', activation='relu')(x4)
	#x5 = layers.BatchNormalization(momentum=0.8)(x5)

	xd = layers.Dense(128)(x)

	x1d = layers.Dense(128)(x1)

	x2d = layers.Dense(128)(x2)

	x3d = layers.Dense(128)(x3)

	x4d = layers.Dense(128)(x4)

	#x5d = layers.Dense(128)(x5)
	#x5d = layers.Dense(128)(x5d)

	xf = tf.concat([xd, x1d, x2d, x3d, x4d], -1)
	xf = layers.Dense(128)(xf)
	xfd = layers.Conv3D(1, 1, strides=(1, 1, 1), padding='same', activation='relu')(xf)

	xfd = tf.keras.activations.tanh(xfd)
	xfd = layers.Reshape((xdim, ydim, zdim))(xfd)
	model = models.Model(inputs=struct, outputs=xfd)

	return model
