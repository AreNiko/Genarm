import glob
import os
import argparse
import time
import csv
from functools import partial
#import matlab.engine

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tensorflow as tf
from tensorflow.keras import losses, metrics, optimizers

import gen_model



def generate_random_struct(dims):
	struct = tf.random.uniform(shape=dims, minval=0, maxval=2, dtype=tf.int32)
	struct = tf.cast(struct, tf.float32)
	#lay1 = [[0,0,0],[0,0,0],[0,0,0]]
	#lay2 = [[0,1,0],[1,1,1],[0,0,0]]
	#lay3 = [[0,0,0],[0,1,1],[0,1,0]]
	return struct
	"""
	struct = np.zeros(dims)
	struct[5:12, 8:12, 8:12] = 1
	struct[7:10, 9:11, 8:12] = 0

	#struct = tf.constant([lay, lay0, lay1, lay2, lay3, lay4, lay5, lay6, lay7])
	return tf.convert_to_tensor(struct,dtype=tf.float32)
	"""

def DSM(x):
	
	return y


def get_optimizer():
	# Just constant learning rate schedule
	optimizer = optimizers.Adam(lr=1e-4)
	return optimizer

def custom_loss_function(new_struct, struct):
	# Apply direct stiffness method as loss function
	#(E, N,_) = eng.vox2mesh18(x);
	#(sE, dN) = eng.FEM_truss(N,E, extF,extC)
	#loss = max(abs(dN))
	"""
	x = tf.reduce_sum(struct)
	y = tf.reduce_sum(new_struct)
	#print(x)
	#print(y)
	#loss = tf.losses.mean_squared_error(tf.reduce_sum(new_struct), tf.reduce_sum(struct))
	loss = tf.abs(tf.abs(x)-tf.abs(y))
	print(loss)
	"""

	loss = tf.abs(tf.math.subtract(new_struct, struct))
	loss = tf.reduce_sum(loss)
	return loss

def runstuff(train_dir):
	# Construct model and measurements
	x = 20; y = 20; z = 20
	struct1 = generate_random_struct((x,y,z))
	struct2 = generate_random_struct((x,y,y))
	struct3 = generate_random_struct((x,y,z))
	struct4 = generate_random_struct((x,y,z))

	"""
	struct1 = tf.reshape(struct1, (x,y,z,1))
	struct2 = tf.reshape(struct2, (x,y,z,1))
	struct3 = tf.reshape(struct3, (x,y,z,1))
	struct4 = tf.reshape(struct4, (x,y,z,1))
	"""
	#lines = np.loadtxt("../vG.txt", comments="#", delimiter=",", unpack=False)
	#arr = np.reshape(lines, (100,100,200))

	dataset = tf.data.Dataset.from_tensor_slices([[struct1], [struct2], [struct3], [struct4]])
	print(dataset)
	#mse = tf.keras.losses.MeanSquaredError()
	model = gen_model.Model3D((x,y,z))
	optimizer = get_optimizer()
	#model.compile(optimizer='adam', loss=custom_loss_function)

	train_loss = metrics.Mean()

	model.summary()

	def train_step(struct):
		"""Updates model parameters, as well as `train_loss` and `train_accuracy`

		Args:
			3D matrix: float tensor of shape [batch_size, height, width, 3]

		Returns:
			3D matrix [batch_size, height, width, 1] with probabilties
		"""

		# TODO: Implement
		with tf.GradientTape() as g:

			new_struct = model(struct, training=True)
			
			# Calculate loss adn accuracy of prediction
			loss = custom_loss_function(new_struct, struct)
			#loss = mse(struct, new_struct)

		print(loss.numpy())
		grad = g.gradient(loss, model.trainable_weights)
		# Calculate gradient and update model
		optimizer.apply_gradients(zip(grad, model.trainable_weights))
		
		# Update loss and accuracy
		train_loss.update_state(loss)

		return new_struct

	# Test model
	#print(struct1)
	#print(tf.shape(struct1))


	colors = np.empty([x,y,z] + [4], dtype=np.float32)
	alpha = .8
	for i in range(x):
		colors[i] = [1-(i/x), 1-((i*i)/(x*x)), (i*i*i)/(x*x*x), alpha]
	#colors[1] = [0, 1, 0, alpha]
	#colors[2] = [0, 0, 1, alpha]
	#colors[3] = [1, 1, 0, alpha]
	#colors[4] = [0, 1, 1, alpha]
	#colors[5] = [1, 1, 1, alpha]

	out = model(struct1)
	#print(out)
	outnp = out.numpy()
	print(tf.shape(struct1))
	print(tf.shape(out))
	
	print(struct1[10])
	print(out[10])
	print(tf.reduce_sum(tf.abs(tf.math.subtract(out[10], struct1[10]))))
	fig0 = plt.figure()
	ax0 = fig0.add_subplot(111, projection='3d')
	ax0.voxels(np.rot90(struct1.numpy(), k=-1, axes=(0, 2)), facecolors=colors, edgecolors='grey')
	ax0.set_xlim3d(-1, x+1)
	ax0.set_ylim3d(-1, y+1)
	ax0.set_zlim3d(-1, z+1)
	ax0.set_xlabel('x')
	ax0.set_ylabel('y')
	ax0.set_zlabel('z')

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.voxels(np.rot90(outnp, k=-1, axes=(0, 2)), facecolors=colors, edgecolors='grey')
	ax.set_xlim3d(-1, x+1)
	ax.set_ylim3d(-1, y+1)
	ax.set_zlim3d(-1, z+1)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

	#plt.show()
	#plt.savefig("demo.png")
	

	#eng = matlab.engine.start_matlab()
	
	print("Summaries are written to '%s'." % train_dir)
	train_writer = tf.summary.create_file_writer(
		os.path.join(train_dir, "train"), flush_millis=3000)
	val_writer = tf.summary.create_file_writer(
		os.path.join(train_dir, "val"), flush_millis=3000)
	summary_interval = 10

	step = 0
	start_training = start = time.time()
	for epoch in range(500):
		print("Training epoch: %d" % epoch)
		#for element in dataset:
		new_struct = train_step(struct1)
		"""
		out = new_struct.numpy()
		out[out < 0.1] = 0
		fig1= plt.figure()
		ax1 = fig1.add_subplot(111, projection='3d')
		ax1.voxels(np.rot90(np.ceil(out), k=-1, axes=(0, 2)), facecolors=colors, edgecolors='grey')
		ax1.set_xlim3d(-1, x+1)
		ax1.set_ylim3d(-1, y+1)
		ax1.set_zlim3d(-1, z+1)
		ax1.set_xlabel('x')
		ax1.set_ylabel('y')
		ax1.set_zlabel('z')

		plt.show()
		"""
		#plt.pause(1)
		#struct1 = new_struct
		
		step += 1

		# summaries to terminal
		"""
		if step % summary_interval == 0:
			duration = time.time() - start
			print("step %3d. sec/batch: %.3f. Train loss: %g" % (
				step, duration/summary_interval, train_loss.result().numpy()))
			start = time.time()
				
		# write summaries to TensorBoard
		with train_writer.as_default():
			tf.summary.scalar("loss", train_loss.result(), step=epoch+1)
		#	tf.summary.scalar("accuracy", train_accuracy.result(), step=epoch+1)
		#	vis = vis_mask(image, y_pred >= 0.5)
		#	tf.summary.image("train_image", vis, step=epoch+1)

		# reset metrics
		train_loss.reset_states()
		"""
		
	out = new_struct.numpy()
	out[out < 0.1] = 0
	fig1= plt.figure()
	ax1 = fig1.add_subplot(111, projection='3d')
	ax1.voxels(np.rot90(np.ceil(out), k=-1, axes=(0, 2)), facecolors=colors, edgecolors='grey')
	ax1.set_xlim3d(-1, x+1)
	ax1.set_ylim3d(-1, y+1)
	ax1.set_zlim3d(-1, z+1)
	ax1.set_xlabel('x')
	ax1.set_ylabel('y')
	ax1.set_zlabel('z')

	plt.show()

	
	

def parse_args():
  """Parse command line argument."""

  parser = argparse.ArgumentParser("Train segmention model on Kitti dataset.")
  parser.add_argument("train_dir", help="Directory to put logs and saved model.")

  return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()
	runstuff(args.train_dir)
