import glob
import os
import argparse
import time
import csv
from functools import partial
import matlab
import matlab.engine


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



def get_optimizer():
	# Just constant learning rate schedule
	optimizer = optimizers.Adam(lr=1e-4)
	return optimizer

def custom_loss_function(true_struct, new_struct):
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

	diff = tf.abs(tf.math.subtract(new_struct, true_struct))
	loss = tf.reduce_sum(diff)
	return loss

def plot_vox(fig, struct, colors, dims, show=False):
	ax = fig.add_subplot(111, projection='3d')
	ax.voxels(np.rot90(struct, k=-1, axes=(0, 2)), facecolors=colors, edgecolors='grey')
	ax.set_xlim3d(-1, dims[0]+1)
	ax.set_ylim3d(-1, dims[1]+1)
	ax.set_zlim3d(-1, dims[2]+1)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

	if show:
		plt.show()

def runstuff(train_dir, use_mlab):
	# Construct model and measurements
	
	if use_mlab == "1":
		eng = matlab.engine.start_matlab()
		"""
		names = matlab.engine.find_matlab()
		print(names)
		if not names:
			eng = matlab.engine.start_matlab()
		else:
			eng = matlab.engine.connect_matlab(names[0])
		"""
		struct1og = eng.get_struct1()
		print(type(struct1og))
		struct1 = np.array(struct1og)
		(x,y,z) = struct1.shape
		struct1 = tf.convert_to_tensor(struct1, dtype=tf.float32)
		eng.plotVg_safe(struct1og, 'edgeOff', nargout=0)

	else:
		x = 20; y = 20; z = 20
		struct1 = generate_random_struct((x,y,z))
		struct2 = generate_random_struct((x,y,y))
		struct3 = generate_random_struct((x,y,z))
		struct4 = generate_random_struct((x,y,z))
		dataset = tf.data.Dataset.from_tensor_slices([[struct1], [struct2], [struct3], [struct4]])

	print(x,y,z)
	model = gen_model.Model3D((x,y,z))
	optimizer = get_optimizer()
	#model.compile(optimizer='adam', loss=custom_loss_function)

	train_loss = metrics.Mean()

	model.summary()

	def train_step(struct, true_struct):
		"""Updates model parameters, as well as `train_loss` and `train_accuracy`

		Args:
			3D matrix: float tensor of shape [batch_size, height, width, depth]

		Returns:
			3D matrix [batch_size, height, width, depth] with probabilties
		"""

		# TODO: Implement
		with tf.GradientTape() as g:

			new_struct = model(struct, training=True)
			
			# Calculate loss and accuracy of prediction
			loss = custom_loss_function(true_struct, new_struct)
			#loss = mse(true_struct, new_struct)

		#print(loss.numpy())
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


	out = model(struct1)
	struct1ml = matlab.int8(np.int8(struct1.numpy()).tolist())
	eng.plotVg_safe(struct1ml, 'edgeOff', nargout=0)

	outml = matlab.int8(np.int8(out.numpy()).tolist())
	eng.plotVg_safe(outml, 'edgeOff', nargout=0)
	#fig0 = plt.figure()
	#plot_vox(fig0, struct1.numpy(), colors, [x,y,z])

	#fig = plt.figure()
	#plot_vox(fig, out.numpy(), colors, [x,y,z], True)

	#plt.savefig("demo.png")
	
	
	print("Summaries are written to '%s'." % train_dir)
	train_writer = tf.summary.create_file_writer(
		os.path.join(train_dir, "train"), flush_millis=3000)
	val_writer = tf.summary.create_file_writer(
		os.path.join(train_dir, "val"), flush_millis=3000)
	summary_interval = 10

	step = 0
	start_training = start = time.time()
	for epoch in range(500):
		# Trains model on structures with a truth structure created from
		# The direct stiffness method and shifted voxels
		true_struct = eng.reinforce_struct(matlab.int8(np.int8(struct1.numpy()).tolist()))
		true_struct = np.array(true_struct)
		true_struct = tf.convert_to_tensor(true_struct, dtype=tf.float32)
		new_struct = train_step(struct1, true_struct)

		struct1 = true_struct 
		"""
		for element in dataset:
			new_struct = train_step(element, true_element)
		"""

		step += 1

		# summaries to terminal
		if epoch % summary_interval == 0:
			#print("Training epoch: %d" % epoch)
			print("Epoch %3d. Train loss: %f" % (epoch, train_loss.result().numpy()))
			out = new_struct.numpy()
			out[out < 0.1] = 0
			eng.plotVg_safe(matlab.int8(np.int8(out.numpy()).tolist()), 'edgeOff', nargout=0)
		
		# write summaries to TensorBoard
		with train_writer.as_default():
			tf.summary.scalar("loss", train_loss.result(), step=epoch+1)

		# reset metrics
		train_loss.reset_states()
				
	out = new_struct.numpy()
	out[out < 0.1] = 0
	eng.plotVg_safe(matlab.int8(np.int8(out.numpy()).tolist()), 'edgeOff', nargout=0)
	#fig1= plt.figure()
	#plot_vox(fig1, out.numpy(), colors, [x,y,z], True)
	

def parse_args():
  """Parse command line argument."""

  parser = argparse.ArgumentParser("Train segmention model on 3D structures.")
  parser.add_argument("train_dir", help="Directory to put logs and saved model.")
  parser.add_argument("use_mlab", help="1 or 0, activate matlab session for generating structure.")

  return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()
	runstuff(args.train_dir, args.use_mlab)
