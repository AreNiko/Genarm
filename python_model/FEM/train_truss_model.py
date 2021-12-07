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

import truss_model

def runstuff(train_dir, continue_train=True):
	# Construct model and measurements
	collist = matlab.double([0, 0.68, 0.7647])
	batch_size = 1

	#for struct, true_struct in new_dataset:
		#print(tf.shape(struct))
		#print(tf.shape(true_struct))
		
	names = matlab.engine.find_matlab()
	#print(names)
	if not names:
		eng = matlab.engine.start_matlab()
	else:
		eng = matlab.engine.connect_matlab(names[0])

	structog, vgc, vGextC, vGextF, vGstayOff = eng.get_struct1(nargout=5)
	E,N,_ = eng.vox2mesh18(structog,nargout=3)
	extC = vGextC2extC(vGextC,structog,nargout=1)
	extF = vGextF2extF(vGextF,structog,nargout=1)
	structC = np.array(vGextC)
	extC[:,2] = 0
	structF = np.array(vGextF)
	extF = extF*np.array([0, 1, 0])
	
	E_arr = np.array(E)
	N_arr = np.array(N)
	structStayOff = np.array(vGstayOff)
	print(E_arr.shape)
	print(N_arr.shape)
	struct = np.array(structog)
	(x,y,z) = structStayOff.shape

	#struct = tf.stack([struct, structC, structF, structOff], -1)
	new_dataset = tf.data.Dataset.from_tensors(E,N, structC, structF, vGstayOff)
	new_dataset = new_dataset.batch(batch_size)
	eng.plotVg_safe(structog, 'edgeOff', nargout=0)
		
	model = truss_model.trussStructModel(E_arr.shape, N_arr.shape, structC.shape, structF.shape (x,y,z))
	optimizer = get_optimizer()
	mse = losses.MeanSquaredError()
	train_loss = metrics.Mean()
	model.summary()

	def train_step(input1, input2, structC, structF, vGstayOff, true_struct):
		"""Updates model parameters, as well as `train_loss`
		Args:
			3D matrix: float tensor of shape [batch_size, height, width, depth]

		Returns:
			3D matrix [batch_size, height, width, depth] with probabilties
		"""
		with tf.GradientTape() as g:

			new_struct = model(struct, training=True)
			
			# Calculate loss and accuracy of prediction
			loss = truss_loss_function(true_struct, new_struct, struct)

		grad = g.gradient(loss, model.trainable_weights)
		# Calculate gradient and update model
		optimizer.apply_gradients(zip(grad, model.trainable_weights))
		
		# Update loss and accuracy
		train_loss.update_state(loss)

		return new_struct

	
	def truss_loss_function(E1,N1, E,N, extF, extC):
		# Apply direct stiffness method as loss function
		#sE, dN = eng.Struct_bend(matlab.int8(np.int8(np.ceil(new_struct[0])).tolist()), extC, extF, nargout=2)
		#sE2, dN2 = eng.Struct_bend(matlab.int8(np.int8(np.ceil(struct[0])).tolist()), extC, extF, nargout=2)

		#E1,N1 = vox2mesh18(new_struct)
		#radius = 0.003
		#E1[:,2] = np.pi*radius**2

		#extC1 = vGextC2extC(extC,new_struct[0])
		#extF1 = vGextF2extF(extF,new_struct[0])
		#extF1 = extF1*np.array([0, 1, 0])
		#extC1[:,2] = 0
		
		sE, dN = FEM_truss(E1,N1,extF,extC)
		#print(np.array(dN).max())
		#E2,N2 = vox2mesh18(struct)
		#E2[:,2] = np.pi*radius**2
	
		#extC2 = vGextC2extC(extC,struct[0])
		#extF2 = vGextF2extF(extF,struct[0])
		#extF2 = extF2*np.array([0, 1, 0])
		#extC2[:,2] = 0
		
		#sE2, dN2 = FEM_truss(E2,N2,extF2,extC2)
		
		#print(np.array(dN2).max())
		#loss = (np.array(dN2).max()-np.array(dN).max())*tf.losses.mean_squared_error(np.array(new_struct), np.array(struct))
		#loss = tf.reduce_sum(new_struct) - tf.reduce_sum(struct)
		loss = tf.reduce_sum(E2) - tf.reduce_sum(E1) 
		return loss

	train = train_step
	checkpoint_path = "training_trussModel/cp-{epoch:04d}.ckpt"


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

	summary_interval = 5

	step = 0
	avg_loss_vals = []
	step_loss_vals = []
	
	for epoch in range(100):
		# Trains model on structures with a truth structure created from
		# The direct stiffness method and shifted voxels
			avg_diff = 0
			avg_loss = 0
			datlen = 0
			for inpu, true_struct in new_dataset:
				new_struct = train(tf.cast(inpu,tf.float32), tf.cast(true_struct,tf.float32))

				avg_loss += train_loss.result().numpy()
				datlen += 1
				step += 1
				step_loss_vals.append([step, train_loss.result().numpy()])

				print("Train loss: %f" % (train_loss.result().numpy()))	

			avg_loss /= datlen
			avg_loss_vals.append([epoch, avg_loss])
			print("Avg Train loss: %f" % (avg_loss))

		# summaries to terminal
		if epoch % summary_interval == 0:
			#print("Training epoch: %d" % epoch)
			model.save_weights(checkpoint_path.format(epoch=epoch))
			print("Epoch %3d. Train loss: %f" % (epoch, train_loss.result().numpy()))

			out = new_struct.numpy()
			out[out <= 0.05] = 0
			out[out > 0.05] = 1
		
		# write summaries to TensorBoard
		with train_writer.as_default():
			tf.summary.scalar("loss", train_loss.result(), step=epoch+1)

		# reset metrics
		train_loss.reset_states()
	
	with open("Truss_avg_loss.txt", "wb") as fp:
		pickle.dump(avg_loss_vals, fp)
	with open("Truss_step_loss.txt", "wb") as fp:
		pickle.dump(step_loss_vals, fp)


def parse_args():
  """Parse command line argument."""
  parser = argparse.ArgumentParser("Train segmention model on 3D structures.")
  parser.add_argument("train_dir", help="Directory to put logs and saved model.")

  return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()
	use_mlab 		= True
	train_reinforce = True
	continue_train 	= False
	runstuff(args.train_dir, use_mlab, train_reinforce, continue_train)
