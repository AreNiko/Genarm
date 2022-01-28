import glob
import os
import argparse
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

import tensorflow as tf
from tensorflow.keras import losses, metrics, optimizers, layers, Sequential
from tensorflow.keras.layers.experimental import preprocessing

import gen_model
from FEM.vox2mesh18_tensor import vox2mesh18
from FEM.FEM_truss import FEM_truss
from FEM.vGextC2extC import vGextC2extC
from FEM.vGextF2extF import vGextF2extF

def get_data(folder, test_number, filename):
    path = folder + test_number
    if not os.path.exists(path):
        os.makedirs(path)
       
    if not os.path.isfile(path+filename):
        print("No file named: " + path + filename)
        open(path+filename, "w+")

    if os.path.getsize(path+filename) == 0:
        out = []
    else:
        with open(path+filename, "rb") as fp:
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
	#new_struct = tf.math.divide(tf.math.add(tf.math.subtract(new_struct, m),tf.math.abs(tf.math.subtract(new_struct, m))), tf.math.multiply(2,tf.math.abs(tf.math.subtract(new_struct, m))))
	true_struct2 = tf.math.subtract(struct, true_struct)
	new_struct2 = tf.math.subtract(struct, new_struct)
	#diff = tf.abs(tf.math.subtract(new_struct2, true_struct2))
	#loss = tf.reduce_sum(diff)
	loss = tf.losses.mean_squared_error(true_struct2, new_struct2) 
	return loss

def convert_to_matlabint8(inarr):
	return matlab.int8(np.int8(np.ceil(inarr)).tolist())

def description(loc):
	print("Please describe the purpose and difference of this experiment:")
	desc = input()
	description = open(loc + "experiment_description.txt","w")
	description.write(desc)
	description.close()
	print("Description has been saved to: " + loc + "experiment_description.txt")

def runstuff(train_dir, test_number, use_pre_struct=True, continue_train=True, model_type=2):

	# Construct model and measurements
	
	batch_size = 1

	trainAug = Sequential([
	layers.RandomFlip(mode="horizontal_and_vertical"),
	layers.RandomRotation(0.25)
	])

	if use_pre_struct:
		import matlab
		import matlab.engine
		
		collist = matlab.double([0, 0.68, 0.7647])
		names = matlab.engine.find_matlab()
		#print(names)
		if not names:
			eng = matlab.engine.start_matlab()
		else:
			eng = matlab.engine.connect_matlab(names[0])

		structog, vGextC, vGextF, vGstayOff = eng.get_struct2(nargout=4)
		
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

	else:
		path = os.path.abspath(os.getcwd()) + "/data/reinforce_all/004" 
		new_dataset = tf.data.experimental.load(path)
		data_size = tf.data.experimental.cardinality(new_dataset).numpy()
		new_dataset = new_dataset.batch(batch_size)
		new_dataset = new_dataset.shuffle(data_size+1)

		train_size = int(0.7 * data_size)
		val_size = int(0.15 * data_size)
		test_size = int(0.15 * data_size)

		train_dataset = new_dataset.take(train_size)
		val_dataset = new_dataset.skip(train_size).take(val_size)
		test_dataset = new_dataset.skip(train_size).skip(val_size)

		for struct, structC, structF, stayoff, true_struct in train_dataset.take(1):
			(xstruct,ystruct,zstruct) = struct[0].numpy().shape
			(xC,yC,zC) = structC[0].numpy().shape
			(xF,yF,zF) = structF[0].numpy().shape
			(xoff,yoff,zoff) = stayoff[0].numpy().shape
	#dims = tf.stack([, (xC,yC,zC), (xF,yF,zF), (xoff,yoff,zoff)])

	if model_type == 0:
		model = gen_model.Model3D((xstruct,ystruct,zstruct))
	elif model_type == 1:
		model = gen_model.ConvModel3D((xstruct,ystruct,zstruct), (xC,yC,zC), (xF,yF,zF), (xoff,yoff,zoff))
	else:
		model = gen_model.ConvStructModel3D((xstruct,ystruct,zstruct, 4))

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
		#m = tf.Variable(0.5)
		

		inpus = tf.stack([struct, structC, structF, stayoff], axis=4)
		with tf.GradientTape() as g:
			if model_type == 0:
				new_struct = model(struct, training=istrain)
			elif model_type == 1:
				new_struct = model([struct, structC, structF, stayoff], training=istrain)
			else:
				new_struct = model(inpus, training=istrain)
			
			# Calculate loss and accuracy of prediction
			loss = custom_loss_function(true_struct, new_struct, struct)
		
		# Calculate gradient, update model, and Update loss
		grad = g.gradient(loss, model.trainable_weights)

		if istrain:
			optimizer.apply_gradients(zip(grad, model.trainable_weights))
			#optimizer.apply_gradients([(accum_vars[i], tv) for i, tv in enumerate(model.trainable_weights)])
			train_loss.update_state(loss)
		else:
			val_loss.update_state(loss)

		return new_struct, grad

	model_folder = "models_" + str(model_type) + "/"
	result_folder = "results_" + str(model_type) + "/"
	os.makedirs(os.path.dirname(model_folder + test_number + "/"), exist_ok=True)
	os.makedirs(os.path.dirname(result_folder + test_number + "/"), exist_ok=True)
	checkpoint_path = model_folder + "checkpoints/" + test_number + "/cp-{epoch:04d}.ckpt"
	checkpoint_dir = os.path.dirname(checkpoint_path)
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

	avg_diff_vals = []
	avg_loss_vals = []
	avg_val_diff_vals = []
	avg_val_loss_vals = []

	step_diff_vals = []
	step_loss_vals = []
	step_val_diff_vals = []
	step_val_loss_vals = []

	step = 0
	step_val = 0
	saved_progress = 0

	if continue_train:
		step_loss_vals = get_data(result_folder, test_number, "/step_loss.txt")
		step_diff_vals = get_data(result_folder, test_number, "/step_diff.txt")
		if step_loss_vals:
			step = step_loss_vals[-1][0]+1
		else:
			step = 0

		avg_loss_vals = get_data(result_folder, test_number, "/avg_train_loss.txt")
		avg_diff_vals = get_data(result_folder, test_number, "/avg_train_diff.txt")
		if avg_loss_vals:		
			saved_progress = avg_loss_vals[-1][0]+1
		else:
			saved_progress = 0
		
		step_val_loss_vals = get_data(result_folder, test_number, "/step_val_loss.txt")
		step_val_diff_vals = get_data(result_folder, test_number, "/step_val_diff.txt")
		if step_val_loss_vals:
			step = step_val_loss_vals[-1][0]+1
		else:
			step = 0
			
		avg_val_loss_vals = get_data(result_folder, test_number, "/avg_val_loss.txt")
		avg_val_diff_vals = get_data(result_folder, test_number, "/avg_val_diff.txt")

		try:
			latest = tf.train.latest_checkpoint(checkpoint_dir)
			model.load_weights(latest)

		except AttributeError:
			print("No previously saved weights")

	print("Summaries are written to '%s'." % train_dir)
	train_writer = tf.summary.create_file_writer(
		os.path.join(train_dir, "train"), flush_millis=3000)

	summary_interval_step = 50
	summary_interval = 1
	
	tol_list = []
	tol_val_list = []
	avg_tol = 0
	avg_tol_val = 0
	tol = np.linspace(0.0, 1.0, 101)

	print("Starting from epoch: %d | On training step: %d | Validation step: %d" % (saved_progress, step, step_val))
	for epoch in range(saved_progress, 10):
		# Trains model on structures with a truth structure created from
		# The direct stiffness method and shifted voxels
		if use_pre_struct:
			for inpu, structC, structF, structOff in new_dataset.take(1):
				new_struct = train_matlab(inpu, structC, structF, structOff)
			out = new_struct.numpy()
			out[out <= 0.05] = 0
			out[out > 0.05] = 1
			eng.clf(nargout=0)
			eng.plotVg_safe(matlab.int8(np.int8(np.ceil(out)).tolist()), 'edgeOff', 'col',collist, nargout=0)
			step += 1
			print("Train loss: %f" % (train_loss.result().numpy()))
		else:
			avg_diff = 0.0
			avg_loss = 0.0
			datlen = 0
			
			for inpu, structC, structF, stayoff, true_struct in train_dataset:
				new_struct, grad = train_step(tf.cast(inpu,tf.float32), tf.cast(structC,tf.float32), tf.cast(structF,tf.float32), tf.cast(stayoff,tf.float32), tf.cast(true_struct,tf.float32), True)	
				
				# Calculate gradient, update model, and Update loss
				#tvs = model.trainable_weights
				#accum_vars = [tf.Variable(tv.initialized_value(), trainable=False) for tv in tvs]
				#zero_ops  = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
				#accum_ops = [accum_vars[i].assign_add(gv) for i, gv in enumerate(grad)]

				#print(tvs)
				#if step % 5 == 0:
					#optimizer.apply_gradients(zip(grad, model.trainable_weights))
					#optimizer.apply_gradients([(accum_vars[i], tv) for i, tv in enumerate(model.trainable_weights)])

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
					
				print("Step: %d | Training Difference: %d | Train loss: %f | Current tol: %.2f" % (step, check_diff, train_loss.result().numpy(), current_tol))	
				#print("Step: %d | Training Difference: %d | Train loss: %f " % (step, check_diff, train_loss.result().numpy()))	

				# Records the tolerance and the stepwise measurements
				avg_diff += check_diff 
				avg_loss += train_loss.result().numpy()

				step_diff_vals.append([step, check_diff])
				step_loss_vals.append([step, train_loss.result().numpy()])
				
				if step % summary_interval_step == 0:
					#print("Training step: %d" % step)
					with open(result_folder + test_number + "/step_loss.txt", "wb") as fp:
						pickle.dump(step_loss_vals, fp)
					with open(result_folder + test_number + "/step_diff.txt", "wb") as fp:
						pickle.dump(step_diff_vals, fp)
					model.save_weights(checkpoint_path.format(epoch=epoch))
					#print("Epoch %3d. step %3d. Training loss: %f" % (epoch, step, train_loss.result().numpy()))

				datlen += 1
				step += 1

			# Calculates the averages for the measurements from training
			avg_diff /= datlen
			avg_loss /= datlen
			avg_diff_vals.append([epoch, avg_diff])
			avg_loss_vals.append([epoch, avg_loss])

			#print("Avg Training tol: %f", avg_tol/len(tol_list))
			print("Avg Training Difference: %d | Avg Train loss: %f" % (avg_diff, avg_loss))
			
			with open(result_folder + test_number + "/step_loss.txt", "wb") as fp:
				pickle.dump(step_loss_vals, fp)
			with open(result_folder + test_number + "/step_diff.txt", "wb") as fp:
				pickle.dump(step_diff_vals, fp)
			with open(result_folder + test_number + "/avg_train_loss.txt", "wb") as fp:
				pickle.dump(avg_loss_vals, fp)
			with open(result_folder + test_number + "/avg_train_diff.txt", "wb") as fp:
				pickle.dump(avg_diff_vals, fp)
			
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
			train_dataset = train_dataset.shuffle(train_size+1)

			avg_val_diff = 0.0
			avg_val_loss = 0.0
			datlen_val = 0
			for inpu, structC, structF, stayoff, true_struct in val_dataset:

				new_struct, _ = train_step(tf.cast(inpu,tf.float32), tf.cast(structC,tf.float32), tf.cast(structF,tf.float32), tf.cast(stayoff,tf.float32), tf.cast(true_struct,tf.float32), False)	
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

				avg_tol_val += current_tol
				out = new_struct[0].numpy()
				out[out <= current_tol] = 0
				out[out > current_tol] = 1

				# Records the tolerance and the stepwise measurements
				tol_val_list.append(current_tol)
				avg_val_diff += check_diff 
				avg_val_loss += val_loss.result().numpy()
				
				step_val_diff_vals.append([step_val, check_diff])
				step_val_loss_vals.append([step_val, val_loss.result().numpy()])
					
				print("Step: %d | Validation Difference: %d | Validation loss: %f | Current tol: %.2f" % (step_val, check_diff, val_loss.result().numpy(), current_tol))
			
				if step_val % summary_interval_step == 0:
					with open(result_folder + test_number + "/step_val_loss.txt", "wb") as fp:
						pickle.dump(step_val_loss_vals, fp)
					with open(result_folder + test_number + "/step_val_diff.txt", "wb") as fp:
						pickle.dump(step_val_diff_vals, fp)

				datlen_val += 1
				step_val += 1
				
			# Calculates the averages for the measurements from training
			avg_val_diff /= datlen_val
			avg_val_loss /= datlen_val
			avg_tol_val /= datlen_val
			avg_val_diff_vals.append([epoch, avg_val_diff])
			avg_loss_vals.append([epoch, avg_val_loss])
			
			with open(result_folder + test_number + "/step_val_loss.txt", "wb") as fp:
				pickle.dump(step_val_loss_vals, fp)
			with open(result_folder + test_number + "/step_val_diff.txt", "wb") as fp:
				pickle.dump(step_val_diff_vals, fp)
			with open(result_folder + test_number + "/avg_val_loss.txt", "wb") as fp:
				pickle.dump(avg_val_loss_vals, fp)
			with open(result_folder + test_number + "/avg_val_diff.txt", "wb") as fp:
				pickle.dump(avg_val_diff_vals, fp)

			print("Avg Validation Difference: %d | Avg Validation loss: %f | Avg Validation tol: %f" % (avg_val_diff, avg_val_loss, avg_tol_val))		
					
			val_dataset = val_dataset.shuffle(val_size+1)
		
	
	if use_pre_struct:
		out = new_struct[0].numpy()
		out[out <= 0.05] = 0
		out[out > 0.05] = 1
		eng.plotVg_safe(matlab.int8(np.int8(np.ceil(out)).tolist()), 'edgeOff', 'col',collist, nargout=0)
		input("Press Enter to continue...")

	print("Training and validation have completed.")
	model.save(model_folder + test_number + "/Genmodel")

def parse_args():
	"""Parse command line argument."""
	parser = argparse.ArgumentParser("Train segmention model on 3D structures.")
	parser.add_argument("train_dir", help="Directory to put logs and saved model.")
	parser.add_argument("test_number", help="logs the result files to specific runs")

	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()
	use_pre_struct  = False
	continue_train 	= True
	print("Select Model type:")
	print("Select 0 for single input only structure matrix")
	print("Select 1 for multiple input of all matrix types")
	print("Select 2 for single input with all matrix types")
	#model_type = int(input())
	model_type = 2
	runstuff(args.train_dir, args.test_number, use_pre_struct, continue_train, model_type)
