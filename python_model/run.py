import glob
import os
import argparse
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import matlab
import matlab.engine

import tensorflow as tf
from tensorflow.keras import losses, metrics, optimizers, layers, Sequential
from tensorflow.keras.layers.experimental import preprocessing

import gen_model


def augment_data(dataset):

	return new_data

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
	optimizer = optimizers.Adam(learning_rate=1e-4)
	return optimizer

def custom_loss_function(true_struct, new_struct, struct):
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
	true_struct2 = tf.math.subtract(struct, true_struct)
	new_struct2 = tf.math.subtract(struct, new_struct)
	#diff = tf.abs(tf.math.subtract(new_struct2, true_struct2))
	#loss = tf.reduce_sum(diff)
	loss = tf.losses.mean_squared_error(true_struct2, new_struct2) 
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

def runstuff(train_dir, use_mlab=True, train_reinforce=True, continue_train=True):
	# Construct model and measurements
	collist = matlab.double([0, 0.68, 0.7647])
	batch_size = 1

	trainAug = Sequential([
	layers.RandomFlip(mode="horizontal_and_vertical"),
	layers.RandomRotation(0.25)
	])
	#for struct, true_struct in new_dataset:
		#print(tf.shape(struct))
		#print(tf.shape(true_struct))
		
	names = matlab.engine.find_matlab()
	#print(names)
	if not names:
		eng = matlab.engine.start_matlab()
	else:
		eng = matlab.engine.connect_matlab(names[0])


	if use_mlab:
		
		
		structog, vgc, vGextC, vGextF, vGstayOff = eng.get_struct1(nargout=5)
		struct = np.array(structog)
		(x,y,z) = struct.shape
		struct = tf.convert_to_tensor(struct, dtype=tf.float32)
		structC = np.array(vGextC)
		structC = tf.convert_to_tensor(structC, dtype=tf.float32)
		structF = np.array(vGextF)
		structF = tf.convert_to_tensor(structF, dtype=tf.float32)
		structOff = np.array(vGstayOff)
		structOff = tf.convert_to_tensor(structOff, dtype=tf.float32)

		#struct = tf.stack([struct, structC, structF, structOff], -1)
		new_dataset = tf.data.Dataset.from_tensors(struct)
		new_dataset = new_dataset.batch(batch_size)
		eng.plotVg_safe(structog, 'edgeOff', nargout=0)
		

	else:
		path = os.path.abspath(os.getcwd()) + "/data/reinforce1/002"
		new_dataset = tf.data.experimental.load(path)
		new_dataset = new_dataset.batch(batch_size)
		"""
		new_dataset = (
			new_dataset
			.shuffle(batch_size * 100)
			.batch(batch_size)
			.map(lambda x, y: (trainAug(x), trainAug(y)))
		)
		"""
		for struct, true_struct in new_dataset.take(1):
			(x,y,z) = struct[0].numpy().shape
		"""
		x = 20; y = 20; z = 20
		struct1 = generate_random_struct((x,y,z))
		struct2 = generate_random_struct((x,y,y))
		struct3 = generate_random_struct((x,y,z))
		struct4 = generate_random_struct((x,y,z))
		dataset = tf.data.Dataset.from_tensor_slices([[struct1], [struct2], [struct3], [struct4]])
		"""
		colors = np.empty([x,y,z] + [4], dtype=np.float32)
		alpha = .8
		for i in range(x):
			colors[i] = [1-(i/x), 1-((i*i)/(x*x)), (i*i*i)/(x*x*x), alpha]


	print(x,y,z)
	#model = gen_model.Model3D((x,y,z))
	model = gen_model.ConvStructModel3D((x,y,z))
	optimizer = get_optimizer()

	mse = losses.MeanSquaredError()
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
			loss = custom_loss_function(true_struct, new_struct, struct)
			#loss = mse(true_struct, new_struct)

		#print(loss.numpy())
		grad = g.gradient(loss, model.trainable_weights)
		# Calculate gradient and update model
		optimizer.apply_gradients(zip(grad, model.trainable_weights))
		
		# Update loss and accuracy
		train_loss.update_state(loss)

		return new_struct

	def train_matlab(struct, extF, extC):
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
			loss = truss_loss_function(new_struct, struct, extF, extC)
			#loss = mse(true_struct, new_struct)

		#print(loss.numpy())
		grad = g.gradient(loss, model.trainable_weights)
		# Calculate gradient and update model
		optimizer.apply_gradients(zip(grad, model.trainable_weights))
		
		# Update loss and accuracy
		train_loss.update_state(loss)

		return new_struct

	def train_step_same(struct):
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
			loss = custom_loss_function(struct, new_struct)
			#loss = mse(true_struct, new_struct)

		#print(loss.numpy())
		grad = g.gradient(loss, model.trainable_weights)
		# Calculate gradient and update model
		optimizer.apply_gradients(zip(grad, model.trainable_weights))
		
		# Update loss and accuracy
		train_loss.update_state(loss)

		return new_struct
	
	def truss_loss_function(new_struct, struct, extF, extC):
		# Apply direct stiffness method as loss function
		#(E, N,_) = eng.vox2mesh18(new_struct)
		#(sE, dN) = eng.FEM_truss(N,E,extF,extC)
		sE, dN = eng.Struct_bend(matlab.int8(np.int8(np.ceil(new_struct[0])).tolist()), extC, extF, nargout=2)
		sE2, dN2 = eng.Struct_bend(matlab.int8(np.int8(np.ceil(struct[0])).tolist()), extC, extF, nargout=2)
		print(np.array(dN))		
		loss = (np.array(dN2).max()-np.array(dN).max())*tf.losses.mean_squared_error(np.array(new_struct), np.array(struct))
		#(E2, N2,_) = eng.vox2mesh18(struct)
		#(sE2, dN2) = eng.FEM_truss(N2,E2,extF,extC)
		#loss = max(abs(dN))

		return loss

	if train_reinforce:
		train = train_step
		checkpoint_path = "training_reinforceConv/cp-{epoch:04d}.ckpt"
	else:
		train = train_step_same
		checkpoint_path = "training_duplicateConv/cp-{epoch:04d}.ckpt"

	checkpoint_dir = os.path.dirname(checkpoint_path)
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
	if continue_train:
		latest = tf.train.latest_checkpoint(checkpoint_dir)
		model.load_weights(latest)
	
	print("Summaries are written to '%s'." % train_dir)
	train_writer = tf.summary.create_file_writer(
		os.path.join(train_dir, "train"), flush_millis=3000)
	#val_writer = tf.summary.create_file_writer(os.path.join(train_dir, "val"), flush_millis=3000)
	summary_interval = 5

	step = 0
	avg_diff_vals = []
	avg_loss_vals = []
	step_diff_vals = []
	step_loss_vals = []
	
	#start_training = start = time.time()
	for epoch in range(100):
		# Trains model on structures with a truth structure created from
		# The direct stiffness method and shifted voxels
		if train_reinforce:
			if use_mlab:
				#true_struct = eng.reinforce_struct(matlab.int8(np.int8(struct.numpy()).tolist()), vGextC, vGextF, vGstayOff, 100)
				#eng.plotVg_safe(true_struct, 'edgeOff', nargout=0)
				for inpu in new_dataset.take(1):
					new_struct = train_matlab(inpu, vGextF, vGextC)
				out = new_struct.numpy()
				out[out <= 0.05] = 0
				out[out > 0.05] = 1
				eng.plotVg_safe(matlab.int8(np.int8(np.ceil(out)).tolist()), 'edgeOff', 'col',collist, nargout=0)
				step += 1
				print("Difference: %d | Train loss: %f" % (check_diff, train_loss.result().numpy()))
			else:
				avg_diff = 0
				avg_loss = 0
				datlen = 0
				for inpu, true_struct in new_dataset:
					new_struct = train(tf.cast(inpu,tf.float32), tf.cast(true_struct,tf.float32))
					out = new_struct[0].numpy()
					out[out <= 0.05] = 0
					out[out > 0.05] = 1
					out_true = true_struct[0].numpy()
					check_diff = np.sum(np.abs(out_true - out))
					avg_diff += check_diff 
					avg_loss += train_loss.result().numpy()
					datlen += 1
					step += 1
					step_diff_vals.append([step, check_diff])
					step_loss_vals.append([step, train_loss.result().numpy()])
					eng.plotVg_safe(matlab.int8(np.int8(np.ceil(out)).tolist()), 'edgeOff', 'col',collist, nargout=0)
					print("Difference: %d | Train loss: %f" % (check_diff, train_loss.result().numpy()))	

				avg_diff /= datlen
				avg_loss /= datlen
				avg_diff_vals.append([epoch, avg_diff])
				avg_loss_vals.append([epoch, avg_loss])
				print("Avg Difference: %d | Avg Train loss: %f" % (avg_diff, avg_loss))

				

		else:
			new_struct = train(struct)
			step += 1

		

		# summaries to terminal
		if epoch % summary_interval == 0:
			#print("Training epoch: %d" % epoch)
			model.save_weights(checkpoint_path.format(epoch=epoch))
			print("Epoch %3d. Train loss: %f" % (epoch, train_loss.result().numpy()))

			out = new_struct.numpy()
			out[out <= 0.05] = 0
			out[out > 0.05] = 1
			if use_mlab:
				eng.plotVg_safe(matlab.int8(np.int8(np.ceil(out)).tolist()), 'edgeOff', 'col',collist, nargout=0)
				#eng.plot_struct(matlab.int8(np.int8(np.ceil(out)).tolist()), 3, nargout=0)
			#else:
				#fig2 = plt.figure()
				#plot_vox(fig2, out, colors, [x,y,z], True)
		
		# write summaries to TensorBoard
		with train_writer.as_default():
			tf.summary.scalar("loss", train_loss.result(), step=epoch+1)

		# reset metrics
		train_loss.reset_states()
	
	with open("avg_loss.txt", "wb") as fp:
		pickle.dump(avg_loss_vals, fp)

	with open("avg_diff.txt", "wb") as fp:
		pickle.dump(avg_diff_vals, fp)

	with open("step_loss.txt", "wb") as fp:
		pickle.dump(step_loss_vals, fp)

	with open("step_diff.txt", "wb") as fp:
		pickle.dump(step_diff_vals, fp)

	out = new_struct[0].numpy()
	out[out <= 0.05] = 0
	out[out > 0.05] = 1
	if use_mlab:
		eng.plotVg_safe(matlab.int8(np.int8(np.ceil(out)).tolist()), 'edgeOff', 'col',collist, nargout=0)
		#eng.plot_struct(matlab.int8(np.int8(np.ceil(out)).tolist()), 4, nargout=0)
		input("Press Enter to continue...")
	#else:
		#fig3 = plt.figure()
		#plot_vox(fig3, out, colors, [x,y,z], True)

	
	

def parse_args():
  """Parse command line argument."""

  parser = argparse.ArgumentParser("Train segmention model on 3D structures.")
  parser.add_argument("train_dir", help="Directory to put logs and saved model.")
  #parser.add_argument("use_mlab", help="Set as 1 to activate matlab session for generating structure and plotting.")

  return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()
	use_mlab 		= True
	train_reinforce = True
	continue_train 	= True
	runstuff(args.train_dir, use_mlab, train_reinforce, continue_train)
