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
from FEM.vox2mesh18 import vox2mesh18
from FEM.FEM_truss import FEM_truss
from FEM.vGextC2extC import vGextC2extC
from FEM.vGextF2extF import vGextF2extF

def get_data(filename):
	with open(filename, "rb") as fp:
		out = pickle.load(fp)
	return out

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



def convert_to_matlabint8(inarr):
	return matlab.int8(np.int8(np.ceil(inarr)).tolist())

def runstuff(train_dir, test_number, model_type=2):
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

	#structog, vGextC, vGextF, vGstayOff = eng.get_struct2(nargout=4)
	structog, _, vGextC, vGextF, vGstayOff = eng.get_struct1(14,14,12,100, nargout=5)

	struct = np.array(structog)
	structC = np.array(vGextC)
	structF = np.array(vGextF)
	structOff = np.array(vGstayOff)

	(xstruct,ystruct,zstruct) = struct.shape
	(xC,yC,zC) = structC.shape
	(xF,yF,zF) = structF.shape
	(xoff,yoff,zoff) = structOff.shape

	struct = tf.convert_to_tensor(struct, dtype=tf.float32)
	structCten = tf.convert_to_tensor(structC, dtype=tf.float32)
	structFten = tf.convert_to_tensor(structF, dtype=tf.float32)
	structOfften = tf.convert_to_tensor(structOff, dtype=tf.float32)

	new_dataset = tf.data.Dataset.from_tensors((struct,structCten,structFten,structOfften))
	new_dataset = new_dataset.batch(batch_size)
	eng.plotVg_safe(structog, 'edgeOff', nargout=0)
	
	"""
	#model = gen_model.ConvModel3D((xstruct,ystruct,zstruct), (xC,yC,zC), (xF,yF,zF), (xoff,yoff,zoff))
	model = gen_model.ConvStructModel3D((xstruct,ystruct,zstruct,4))
	optimizer = get_optimizer()
	mse = losses.MeanSquaredError()
	train_loss = metrics.Mean()
	val_loss = metrics.Mean()	
	model.summary()
	#tf.keras.utils.plot_model(model, "3Dconv_model.png", show_shapes=True

	os.makedirs(os.path.dirname("test_results/" + test_number + "/"), exist_ok=True)
	checkpoint_path = "training_reinforce_all2/" + test_number + "/cp-{epoch:04d}.ckpt"
	checkpoint_dir = os.path.dirname(checkpoint_path)
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
	try:
		latest = tf.train.latest_checkpoint(checkpoint_dir)
		model.load_weights(latest)
	except AttributeError:
		print("No trained weights to test on")
	"""

	if model_type == 0:
		model = tf.keras.models.load_model('models_0/' + test_number + '/Genmodel')
	elif model_type == 1:
		model = tf.keras.models.load_model('models_1/' + test_number + '/Genmodel')
	else:
		model = tf.keras.models.load_model('models_2/' + test_number + '/Genmodel')

	def bend_function(struct, vGextC, vGextF):
		sE, dN = eng.Struct_bend(convert_to_matlabint8(struct[0]), convert_to_matlabint8(vGextC[0]), convert_to_matlabint8(vGextF[0]), nargout=2)
		return np.max(np.abs(np.array(dN))), np.sum(np.abs(np.array(dN)))
	
	
	
	
	print("Summaries are written to '%s'." % train_dir)
	train_writer = tf.summary.create_file_writer(
		os.path.join(train_dir, "test"), flush_millis=3000)

	summary_interval_step = 50
	summary_interval = 1
	test_results = []
	compare_results = []
	for struct, structC, structF, structOff in new_dataset.take(1):
		ogtruct = struct
		vG = convert_to_matlabint8(struct)
		lossog = bend_function(struct, structC, structF)
		print('Original max bending: %f | sum bending: %f' % (lossog[0], lossog[1])) 
		for epoch in range(300):
			print("Epoch: ", epoch)
			
			
			if model_type == 0:
				new_struct = model(struct, training=False)
			elif model_type == 1:
				new_struct = model(struct, structC, structF, structOff, training=False)
			else:
				inpus = tf.stack([struct, structC, structF, structOff], axis=4)
				new_struct = model(inpus, training=False)
			
			# Trains model on structures with a truth structure created from
			# The direct stiffness method and shifted vox
			out = new_struct.numpy()
			out[out <= 0.5] = 0
			out[out > 0.5] = 1
			loss = bend_function(out, structC, structF)
			test_results.append([epoch, loss[0], loss[1]])

			structog, sE, dN = eng.reinforce_struct(structog, vGextC, vGextF, vGstayOff, 75, nargout=3)

			compare_results.append([epoch, np.max(np.abs(np.array(dN))), np.sum(np.abs(np.array(dN)))])

			print("3Dconv Model: max bending: %f | sum bending: %f | Amount: %d" % (loss[0], loss[1], np.sum(out)))
			print("Generative  : max bending: %f | sum bending: %f | Amount: %d" % (np.max(np.abs(np.array(dN))), np.sum(np.abs(np.array(dN))), np.sum(np.array(structog))))

			eng.clf(nargout=0)
			eng.plotVg_safe(matlab.int8(np.int8(np.ceil(out[0])).tolist()), 'edgeOff', 'col',collist, nargout=0)
			eng.saveFigToAnimGif('3Dconvoxnet_testing' + test_number + '.gif', epoch==0, nargout=0)
			struct = out
	
	with open("test_results/" + "model" + str(model_type) + "_" + test_number + "/3Dconvmodel_loss.txt", "wb") as fp:
		pickle.dump(test_results, fp)
	with open("test_results/" + "model" + str(model_type) + "_" + test_number + "/generative_loss.txt", "wb") as fp:
		pickle.dump(compare_results, fp)
	with open("test_results/" + "model" + str(model_type) + "_" + test_number + "/3Dconv_structures.txt", "wb") as fp:
		pickle.dump([ogtruct[0].numpy(),out[0]], fp)
	with open("test_results/" + "model" + str(model_type) + "_" + test_number + "/generative_structures.txt", "wb") as fp:
		pickle.dump([ogtruct[0].numpy(),np.array(structog)], fp)

	print('Original max bending: %f | sum bending: %f' % (lossog[0], lossog[1])) 
	print("New max bending: %f | sum bending: %f" % (loss[0], loss[1]))
	eng.clf(nargout=0)
	eng.plotVg_safe(matlab.int8(np.int8(np.ceil(out[0])).tolist()), 'edgeOff', 'col',collist, nargout=0)
	print("Testing have been completed.")
	input("Press Enter to close...")

def parse_args():
	"""Parse command line argument."""
	parser = argparse.ArgumentParser("Train segmention model on 3D structures.")
	parser.add_argument("train_dir", help="Directory to put logs and saved model.")
	parser.add_argument("test_number", help="logs the result files to specific runs")

	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()
	model_type = 0
	runstuff(args.train_dir, args.test_number, model_type)
