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
from vox2mesh18 import vox2mesh18
from FEM_truss_tensor import FEM_truss
from vGextC2extC import vGextC2extC
from vGextF2extF import vGextF2extF


def augment_data(dataset):

	return new_data

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

def convert_vox_to_mesh(struct, vGextF, vGextC):
	E,N = vox2mesh18(struct)
	radius = 0.003
	E[:,2] = np.pi*radius**2

	extC = vGextC2extC(vGextC,struct[0])
	extF = vGextF2extF(vGextF,struct[0])
	extF = extF*np.array([0, 1, 0])
	extC[:,2] = 0
	return E, N, extC, extF

def runstuff(train_dir, use_pre_struct=True, continue_train=True):
	# Construct model and measurements
	collist = matlab.double([0, 0.68, 0.7647])
	batch_size = 1

	trainAug = Sequential([
	layers.RandomFlip(mode="horizontal_and_vertical"),
	layers.RandomRotation(0.25)
	])

	names = matlab.engine.find_matlab()
	#print(names)
	if not names:
		eng = matlab.engine.start_matlab()
	else:
		eng = matlab.engine.connect_matlab(names[0])


	if use_pre_struct:
		structog, vgc, vGextC, vGextF, vGstayOff = eng.get_struct1(nargout=5)
		
		struct = np.array(structog)
		structC = np.array(vGextC)		
		structF = np.array(vGextF)
		structOff = np.array(vGstayOff)
		(x,y,z) = struct.shape

		struct = tf.convert_to_tensor(struct, dtype=tf.float32)
		structCten = tf.convert_to_tensor(structC, dtype=tf.float32)
		structFten = tf.convert_to_tensor(structF, dtype=tf.float32)
		structOfften = tf.convert_to_tensor(structOff, dtype=tf.float32)

		new_dataset = tf.data.Dataset.from_tensors(struct)
		new_dataset = new_dataset.batch(batch_size)
		eng.plotVg_safe(structog, 'edgeOff', nargout=0)

	else:
		path = os.path.abspath(os.getcwd()) + "/data/reinforce_all/002"
		new_dataset = tf.data.experimental.load(path)
		data_size = tf.data.experimental.cardinality(new_dataset).numpy()
		new_dataset = new_dataset.batch(batch_size)
		new_dataset = new_dataset.shuffle(data_size+1)

		train_size = int(0.7 * data_size)
		val_size = int(0.15 * data_size)
		test_size = int(0.15 * data_size)

		train_dataset = new_dataset.take(train_size)
		test_dataset = new_dataset.skip(train_size)
		val_dataset = new_dataset.skip(val_size)
		test_dataset = new_dataset.take(test_size)

		for struct, structC, structF, stayoff, true_struct in train_dataset.take(1):
			(xstruct,ystruct,zstruct) = struct[0].numpy().shape
			(xC,yC,zC) = structC[0].numpy().shape
			(xF,yF,zF) = structF[0].numpy().shape
			(xoff,yoff,zoff) = stayoff[0].numpy().shape

	model = gen_model.ConvModel3D((xstruct,ystruct,zstruct), (xC,yC,zC), (xF,yF,zF), (xoff,yoff,zoff))
	#model = gen_model.ConvStructModel3D((x,y,z))
	optimizer = get_optimizer()
	mse = losses.MeanSquaredError()
	train_loss = metrics.Mean()
	val_loss = metrics.Mean()	
	model.summary()
	#tf.keras.utils.plot_model(model, "3Dconv_model.png", show_shapes=True)

	def train_step(struct, structC, structF, stayoff, true_struct, istrain):
		"""Updates model parameters, as well as `train_loss`
		Args:
			3D matrix: float tensor of shape [batch_size, height, width, depth]

		Returns:
			3D matrix [batch_size, height, width, depth] with probabilties
		"""
		with tf.GradientTape() as g:

			new_struct = model([struct, structC, structF, stayoff], training=istrain)
			
			# Calculate loss and accuracy of prediction
			loss = custom_loss_function(true_struct, new_struct, struct)

		grad = g.gradient(loss, model.trainable_weights)
		# Calculate gradient, update model, and Update loss
		if istrain:
			optimizer.apply_gradients(zip(grad, model.trainable_weights))
			train_loss.update_state(loss)
		else:
			val_loss.update_state(loss)

		return new_struct

	def train_matlab(struct, extF, extC):
		"""Updates model parameters, as well as `train_loss` and `train_accuracy`
		Args:
			3D matrix: float tensor of shape [batch_size, height, width, depth]

		Returns:
			3D matrix [batch_size, height, width, depth] with probabilties
		"""

		with tf.GradientTape() as g:

			new_struct = model(struct, training=True)

			E1, N1, extC1, extF1 = convert_vox_to_mesh(new_struct, extF, extC)
			E2, N2, extC2, extF2 = convert_vox_to_mesh(struct, extF, extC)
			sE, dN = FEM_truss(E1,N1,extF1,extC1)
			sE2, dN2 = FEM_truss(E2,N2,extF2,extC2)
			# Calculate loss and accuracy of prediction
			loss = truss_loss_function(new_struct, struct, extF, extC)

		grad = g.gradient(loss, model.trainable_weights)
		# Calculate gradient and update model
		optimizer.apply_gradients(zip(grad, model.trainable_weights))
		
		# Update loss
		train_loss.update_state(loss)

		return new_struct

	def truss_loss_function(new_struct, struct, extF, extC):
		# Apply direct stiffness method as loss function
		
		#print(np.array(dN2).max())
		#loss = (np.array(dN2).max()-np.array(dN).max())*tf.losses.mean_squared_error(np.array(new_struct), np.array(struct))
		#loss = tf.reduce_sum(new_struct) - tf.reduce_sum(struct)
		loss = tf.math.multiply(tf.reduce_sum(E2) - tf.reduce_sum(E1),tf.reduce_sum(N2) - tf.reduce_sum(N1))
		return loss

	checkpoint_path = "training_reinforce_all/cp-{epoch:04d}.ckpt"
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

	summary_interval_step = 50
	summary_interval = 5

	step = 0
	step_val = 0
	avg_diff_vals = []
	avg_loss_vals = []
	avg_val_diff_vals = []
	avg_val_loss_vals = []

	step_diff_vals = []
	step_loss_vals = []
	step_val_diff_vals = []
	step_val_loss_vals = []
	tol_list = []
	tol_val_list = []
	avg_tol = 0
	avg_tol_val = 0
	tol = np.linspace(0.0, 1.0, 101)

	for epoch in range(100):
		# Trains model on structures with a truth structure created from
		# The direct stiffness method and shifted voxels
		if use_pre_struct:
			for inpu in new_dataset.take(1):
				new_struct = train_matlab(inpu, structF, structC)
			out = new_struct.numpy()
			out[out <= 0.05] = 0
			out[out > 0.05] = 1
			eng.clf(nargout=0)
			eng.plotVg_safe(matlab.int8(np.int8(np.ceil(out)).tolist()), 'edgeOff', 'col',collist, nargout=0)
			step += 1
			print("Train loss: %f" % (train_loss.result().numpy()))
		else:
			avg_diff = 0
			avg_loss = 0
			datlen = 0
			
			for inpu, structC, structF, stayoff, true_struct in train_dataset:
				new_struct = train_step(tf.cast(inpu,tf.float32), tf.cast(structC,tf.float32), tf.cast(structF,tf.float32), tf.cast(stayoff,tf.float32), tf.cast(true_struct,tf.float32), True)	
				out_true = true_struct[0].numpy()
				
				check_diff = 1e23
				for i in tol:
					out = new_struct[0].numpy()
					out[out <= i] = 0
					out[out > i] = 1
						
					check_tol = np.sum(np.abs(out_true - out))
					if check_tol < check_diff:
						check_diff = check_tol
						current_tol = i
				
				avg_tol += current_tol
				tol_list.append(current_tol)
				
				out = new_struct[0].numpy()
				out[out <= current_tol] = 0
				out[out > current_tol] = 1
				check_diff = np.sum(np.abs(out_true - out))
				# Records the tolerance and the stepwise measurements
				avg_diff += check_diff 
				avg_loss += train_loss.result().numpy()
				datlen += 1
				step += 1
				step_diff_vals.append([step, check_diff])
				step_loss_vals.append([step, train_loss.result().numpy()])
					
				print("Step: %d | Training Difference: %d | Train loss: %f | Current tol: %.2f" % (step, check_diff, train_loss.result().numpy(), current_tol))	

				# Plots the current output with the best tolerance
				eng.clf(nargout=0)
				eng.plotVg_safe(matlab.int8(np.int8(np.ceil(out)).tolist()), 'edgeOff', 'col',collist, nargout=0)
				eng.saveFigToAnimGif('3Dconvoxnet_training.gif', step==1, nargout=0)
				
				if step % summary_interval_step == 0:
					#print("Training step: %d" % step)
					model.save_weights(checkpoint_path.format(epoch=epoch))
					print("Epoch %3d. step %3d. Training loss: %f" % (epoch, step, train_loss.result().numpy()))

			# Calculates the averages for the measurements from training
			avg_diff /= datlen
			avg_loss /= datlen
			avg_diff_vals.append([epoch, avg_diff])
			avg_loss_vals.append([epoch, avg_loss])

			#print("Avg Training tol: %f", avg_tol/len(tol_list))
			print("Avg Training Difference: %d | Avg Train loss: %f" % (avg_diff, avg_loss))
			
			avg_val_diff = 0
			avg_val_loss = 0
			datlen_val = 0
			for inpu, structC, structF, stayoff, true_struct in val_dataset:

				new_struct = train_step(tf.cast(inpu,tf.float32), tf.cast(structC,tf.float32), tf.cast(structF,tf.float32), tf.cast(stayoff,tf.float32), tf.cast(true_struct,tf.float32), False)	
				out_true = true_struct[0].numpy()
				check_diff = 1e23
				for i in tol:
					out = new_struct[0].numpy()
					out[out <= i] = 0
					out[out > i] = 1
						
					check_tol = np.sum(np.abs(out_true - out))
					if check_tol < check_diff:
						check_diff = check_tol
						current_tol = i
				print(current_tol)
				avg_tol_val += current_tol
				out = new_struct[0].numpy()
				out[out <= current_tol] = 0
				out[out > current_tol] = 1

				# Records the tolerance and the stepwise measurements
				tol_val_list.append(current_tol)
				avg_val_diff += check_diff 
				avg_val_loss += val_loss.result().numpy()
				datlen_val += 1
				step_val += 1
				step_val_diff_vals.append([step_val, check_diff])
				step_val_loss_vals.append([step_val, val_loss.result().numpy()])
					
				print("Validation Difference: %d | Validation loss: %f" % (check_diff, val_loss.result().numpy()))

				# Plots the current output with the best tolerance
				eng.clf(nargout=0)
				eng.plotVg_safe(matlab.int8(np.int8(np.ceil(out)).tolist()), 'edgeOff', 'col',collist, nargout=0)
				
			# Calculates the averages for the measurements from training
			avg_val_diff /= datlen
			avg_val_loss /= datlen
			avg_val_diff_vals.append([epoch, avg_val_diff])
			avg_loss_vals.append([epoch, avg_val_loss])

			print("Avg Validation tol: %f", avg_tol_val/len(tol_list))	
			print("Avg Validation Difference: %d | Avg Validation loss: %f" % (avg_val_diff, avg_val_loss))				

		# summaries to terminal
		if epoch % summary_interval == 0:
			#print("Training epoch: %d" % epoch)
			model.save_weights(checkpoint_path.format(epoch=epoch))
			#print("Epoch %3d. Validation loss: %f" % (epoch, train_loss.result().numpy()))
		
		# write summaries to TensorBoard
		with train_writer.as_default():
			tf.summary.scalar("loss", train_loss.result(), step=epoch+1)

		# reset metrics
		train_loss.reset_states()
	
	with open("results/avg_loss.txt", "wb") as fp:
		pickle.dump(avg_loss_vals, fp)
	with open("results/avg_diff.txt", "wb") as fp:
		pickle.dump(avg_diff_vals, fp)
	with open("results/step_loss.txt", "wb") as fp:
		pickle.dump(step_loss_vals, fp)
	with open("results/step_diff.txt", "wb") as fp:
		pickle.dump(step_diff_vals, fp)

	with open("results/avg_val_loss.txt", "wb") as fp:
		pickle.dump(avg_val_loss_vals, fp)
	with open("results/avg_val_diff.txt", "wb") as fp:
		pickle.dump(avg_val_diff_vals, fp)
	with open("results/step_val_loss.txt", "wb") as fp:
		pickle.dump(step_val_loss_vals, fp)
	with open("results/step_val_diff.txt", "wb") as fp:
		pickle.dump(step_val_diff_vals, fp)

	
	if use_pre_struct:
		out = new_struct[0].numpy()
		out[out <= 0.05] = 0
		out[out > 0.05] = 1
		eng.plotVg_safe(matlab.int8(np.int8(np.ceil(out)).tolist()), 'edgeOff', 'col',collist, nargout=0)
		input("Press Enter to continue...")

	print("Training and validation have completed.")

def parse_args():
  """Parse command line argument."""
  parser = argparse.ArgumentParser("Train segmention model on 3D structures.")
  parser.add_argument("train_dir", help="Directory to put logs and saved model.")

  return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()
	use_pre_struct  = True
	continue_train 	= False
	runstuff(args.train_dir, use_pre_struct, continue_train)
