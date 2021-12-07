import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def trussStructModel(input1, input2, structC, structF, vGstayOff):

	
	in1x, in1y = input1
	in2x, in2y = input2
	cx, cy = structC
	fx, fy = structF

	input1_shape = (in1x, in1y, 1, 1)
	input2_shape = (in2x, in2y, 1, 1)
	c_shape = (cx, cy, 1, 1)
	f_shape = (fx, fy, 1, 1)
	outxdim, outydim, outzdim = vGstayOff
	stayoff_shape = (outxdim, outydim, outzdim, 1)

	edges = layers.Input(input1_shape)
	nodes = layers.Input(input2_shape)
	locked = layers.Input(c_shape)
	forces = layers.Input(f_shape)
	stayoff = layers.Input(stayoff_shape)

	x1 = layers.Dense(32)(edges)
	x2 = layers.Dense(32)(nodes)
	x3 = layers.Dense(32)(locked)
	x4 = layers.Dense(32)(forces)
	x5 = layers.Conv3D(64, 5, strides=(1, 1, 1), padding='same', activation='relu')(stayoff)

	x1 = layers.Dense(64)(x1)
	x2 = layers.Dense(64)(x2)
	x3 = layers.Dense(64)(x3)
	x4 = layers.Dense(64)(x4)
	x5 = layers.Dense(64)(x5)
	
	xtogether = tf.concat([x1, x2, x3, x4, x5], -1)
	x = layers.Dense(64)(xtogether)
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
	x = tf.keras.activations.tanh(x)
	x = layers.Reshape((outxdim, outydim, outzdim))(x) 
	model = models.Model(inputs=(input1, input2, structC, structF, vGstayOff), outputs=x)

	return model


def trussStructModel3D(input_shape):
	xdim, ydim = input_shape
	input_shape = (xdim, ydim, 1)
	print(input_shape)
	struct = layers.Input(input_shape)

	x1 = layers.Conv3D(128, 5, strides=(1, 1, 1), padding='same', activation='relu')(struct)
	x1 = layers.BatchNormalization(momentum=0.8)(x1)
	
	x2 = layers.Conv3D(64, 5, strides=(1, 1, 1), padding='same', activation='relu')(x1)
	x2 = layers.BatchNormalization(momentum=0.8)(x2)
	
	x3 = layers.Conv3D(32, 5, strides=(1, 1, 1), padding='same', activation='relu')(x2)
	x3 = layers.BatchNormalization(momentum=0.8)(x3)
	
	x4 = layers.Conv3D(16, 5, strides=(1, 1, 1), padding='same', activation='relu')(x3)
	x4 = layers.BatchNormalization(momentum=0.8)(x4)
	

	x1d = layers.Dense(128)(x1)
	x1d = layers.Dense(128)(x1d)

	x2d = layers.Dense(128)(x2)
	x2d = layers.Dense(128)(x2d)

	x3d = layers.Dense(128)(x3)
	x3d = layers.Dense(128)(x3d)

	x4d = layers.Dense(128)(x4)
	x4d = layers.Dense(128)(x4d)

	xf = tf.concat([x1d, x2d, x3d, x4d], -1)
	xfd = layers.Conv3D(1, 1, strides=(1, 1, 1), padding='same', activation='relu')(xf)
	
	xfd = tf.keras.activations.tanh(xfd)
	xfd = layers.Reshape((xdim, ydim, zdim))(xfd) 
	model = models.Model(inputs=struct, outputs=xfd)

	return model
