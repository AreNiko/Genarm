import glob
import os
import argparse
import time
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import matlab
import matlab.engine

import tensorflow as tf
from tensorflow.keras import activations, losses, metrics, optimizers, layers, Sequential
from tensorflow.keras.layers.experimental import preprocessing

from conv3D_networks2 import FeatureExtractor, PolicyNetwork, ValueNetwork

from FEM.vox2mesh18_tensor import vox2mesh18
from FEM.FEM_truss import FEM_truss
from FEM.vGextC2extC import vGextC2extC
from FEM.vGextF2extF import vGextF2extF

collist = matlab.double([0, 0.68, 0.7647])
collist2 = matlab.double([0, 0.7647, 0])
collist3 = matlab.double([0.7647, 0, 0])

def start_engine():
	isengine = matlab.engine.find_matlab()
	print(isengine)
	if not isengine:
		eng = matlab.engine.start_matlab()
		print("Starting matlab engine")
	else:
		eng = matlab.engine.connect_matlab(isengine[0])

	return eng

eng = start_engine()

eng.warning('off',"MATLAB:singularMatrix")

def convert_to_matlabint8(inarr):
	return matlab.int8(np.int8(np.ceil(inarr)).tolist())

def eval_policy(obser, agent, maxlen_environment, eval_episodes, action_repeat):
	episodes = []
	og_struct = obser[:,:,:,:,0]
	structC = obser[:,:,:,:,1]
	structF = obser[:,:,:,:,2]
	stayoff = obser[:,:,:,:,3]
	#eng = start_engine()
	xdim, ydim, zdim = tf.cast(tf.shape(og_struct[0]),tf.float32)

	og_bend = eng.check_max_bend(convert_to_matlabint8(obser[0,:,:,:,0]), convert_to_matlabint8(structC[0]), convert_to_matlabint8(structF[0]))
	scores = []
	best_reward = -1000000000
	best_struct = None

	print("Evaluating Agent:")
	for i in range(eval_episodes):
		print("Episode: ", i)
		observation = obser
		rewards = []
		observations = []
		reward = 0
		t = 0
		done = False
		while True:
			#observation = preprocess(observation)
			observations.append(observation)
			action = agent(observation)[0]
			# remove num_samples dimension and batch dimension
			#action = tf.random.categorical(logits, 30)[0]
			#action = tf.math.sigmoid(tf.cast(action,tf.float32))
			print(action)
			#action = tf.cast(tf.reshape(action[0], [50,3]),tf.float32)
			#action = action/150
			#pi_old = activations.softmax(logits)[0]

			for _ in range(action_repeat):
				t += 1
				new_struct = flip_coord(action, observation[:,:,:,:,0])
				
				done = False
				if np.sum(new_struct) == 0:
					reward = -100.0
					#done = True

				elif np.sum(new_struct[0] - observation[0,:,:,:,0]) == 0:
					reward = 0
					done = True

				else:
					try:
						#eng.clf(nargout=0)
						#eng.plotVg_safe(convert_to_matlabint8(new_struct[0]), 'edgeOff', 'col',collist, nargout=0)
						new_bend, comps = eng.check_max_bend(convert_to_matlabint8(new_struct[0]), 
							                                 convert_to_matlabint8(structC[0]), 
							                                 convert_to_matlabint8(structF[0]), nargout=2)
						if new_bend == 0 or np.isnan(new_bend) or np.isinf(new_bend):
							new_bend = -10.0
							#done = True

						vox_diff = np.abs(np.sum(og_struct.numpy()) - np.sum(new_struct))
						place_diff = np.abs(np.sum(og_struct.numpy() - new_struct))
						bend_diff = og_bend/new_bend
						#print(new_bend, vox_diff, comps)
						
						#reward = bend_diff + place_diff/5 - (vox_diff/10 + (comps-1))
						reward = 1000*(bend_diff-1)
						print("old / new bending: ", og_bend, "/", new_bend)
						print("Difference in voxels: ", vox_diff)

						if comps > 1:
							done = True
						
					except:
						comps = eng.check_components(convert_to_matlabint8(new_struct[0]), nargout=1)
						vox_diff = np.abs(np.sum(og_struct.numpy()) - np.sum(new_struct))
						#reward = - (vox_diff/10 + (comps-1))
						reward = 0
						done = True

					if best_reward < reward or best_reward is None:
						best_struct = new_struct
						best_differences = new_struct[0] - og_struct[0].numpy()

				rewards.append(reward)
				print(reward)

				if done:
					break
				if maxlen_environment >= 0 and t == maxlen_environment:
					break
				
			observation = tf.stack([tf.convert_to_tensor(new_struct), structC, structF, stayoff], axis=4)
			if done:
				print("Episode finished after {} timesteps".format(t+1))
				break
			if maxlen_environment >= 0 and t == maxlen_environment:
				break
		score = sum(rewards)
		scores.append(score)

		if i == 0 or score > best_score:
			best_score = score
			best_episode = observations

	#eng.quit()

	return scores, best_episode, best_struct, best_differences

class Agent(tf.keras.models.Model):
	"""Convenience wrapper around policy network, which returns *encoded*
	actions. Useful when running model for inference and evaluation.
	"""

	def __init__(self, policy_network, **kwargs):
		super(Agent, self).__init__(**kwargs)

		self.policy_network = policy_network

	def call(self, inpu):
		index = self.policy_network(inpu)
		#action = self.action_encoder.index2action(index)
		return index


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

def calculate_returns(rewards, gamma):
	"""Calculate returns, i.e. sum of future discounted rewards, for episode.

	Args:
		rewards : array of shape [sequence_length], with elements
			being the immediate rewards [r_1, r_2, ..., r_T]
		gamma : discount factor, scalar value in [0, 1)

	Returns:
		returns: array of return for each time step, i.e. [g_0, g_1, ... g_{T-1}]
	"""

	returns = np.zeros(len(rewards), dtype=np.float32)
	returns[-1] = rewards[-1]
	for t in reversed(range(1, len(returns))):
		returns[t-1] = rewards[t-1] + gamma*returns[t]

	return returns

def value_loss(target, prediction):
	"""Calculate mean squared error loss for value function predictions.

	Args:
		target : tensor of shape [batch_size] with corresponding target values
		prediction : tensor of shape [batch_size] of predictions from value network

	Returns:
		loss : mean squared error difference between predictions and targets
	"""
	loss = tf.reduce_mean((prediction - target)**2)
	#print("Value: ", loss)
	return loss
def policy_loss(pi_a, pi_old_a, advantage, epsilon):
	"""Calculate policy loss as in https://arxiv.org/abs/1707.06347

	Args:
		pi_a : Tensor of shape [batch_size] with probabilities for actions under
			the current policy
		pi_old_a : Tensor of shape [batch_size] with probabilities for actions
			under the old policy
		advantage : Tensor of shape [batch_size] with estimated advantges for
			the actions (under the old policy)
		epsilon : clipping parameter, float value in (0, 1)

	Returns:
		loss : scalar loss value
	"""

	A = advantage

	# Note: pi_old_a are necessarily non-zero as it was sampled
	f = pi_a / pi_old_a
	f = np.nan_to_num(f, posinf=10)
	#print("Check policy: ", f)

	f_clipped = tf.clip_by_value(f, 1-epsilon, 1+epsilon)
	loss = -tf.minimum(f*A, f_clipped*A)
	# [batch_sizes] ==> scalar
	loss = tf.reduce_mean(loss)
	#print("Policy: ", loss)

	return loss

def estimate_improvement(pi_a, pi_old_a, advantage, t, gamma):
	"""Calculate sample contributions to estimated improvement, ignoring changes to
	state visitation frequencies.

	Args:
		pi_a : Tensor of shape [batch_size] with probabilties for actions under
			the current policy
		pi_old_a : Tensor of shape [batch_size] with probabilities for actions
			under the old policy
		advantage : Tensor of shape [batch_size] with estimated advantges for
			the actions (under the old policy)
		t : Tensor of shape [batch_size], time step fo
		gamma : discount factor scalar value in [0, 1)
	Returns:
		Tensor of shape [batch_size] with estimated sample "contributions" to
		policy improvement.

	Note: in theory advantages*gamma^t should be close to zero, but may be
	different due to randomness or errors in our estimation. Subtracting this
	term seems to make more sense than not doing it.
	"""

	return (pi_a / pi_old_a - 1)*advantage*gamma**t

def estimate_improvement_lb(pi_a, pi_old_a, advantage, t, gamma, epsilon):
	"""Estimate sample contributions to lower bound for improvement, ignoring
	changes to state visitation frequencies.

	Args:
		pi_a : Tensor of shape [batch_size] with probabilities for actions under
			the current policy
		pi_old_a : Tensor of shape [batch_size] with probabilities for actions
			under the old policy
		advantage : Tensor of shape [batch_size] with estimated advantges for
			the actions (under the old policy)
		epsilon : clipping parameter, float value in (0, 1)

	Note: in theory advantages*gamma^t should be close to zero, but may be
	different due to randomness or errors in our estimation. Subtracting this
	term seems to make more sense than not doing it.

	Returns:
		Tensor of shape [batch_size] with estimated sample "contributions" to
		lower bound of policy improvement.

	"""

	A = advantage
	f = pi_a / pi_old_a
	f_clipped = tf.clip_by_value(f, 1-epsilon, 1+epsilon)
	return tf.minimum(f*A, f_clipped*A)*gamma**t - A*gamma**t

def entropy(p):
	"""Entropy base 2, for each sample in batch."""

	log_p = tf.math.log(p + 1e-6) # add small number to avoid log(0)
	# use log2 for easier intepretation
	log2_p = log_p / np.log(2)

	# [batch_size x num_action] --> [batch_size]
	entropy = -tf.reduce_sum(p*log2_p, axis=-1)
	return entropy

def entropy_loss(pi):
	"""Calculate entropy loss for action distributions.

	Args:
		pi : Tensor of shape [batch_size, num_actions], each element in the
			batch is a probability distribution over actions, conditoned on a
			state.

	Returns:
		scalar, average negative entropy for the distributions
	"""
	loss = -tf.reduce_mean(entropy(pi))
	#print("Entropy: ", loss)
	return loss

def flip_coord(action, struct):
	new_struct = struct.numpy()
	batch, xdim, ydim, zdim = tf.shape(struct)
	x = np.floor(action[:,0]*tf.cast(xdim,tf.float32))
	y = np.floor(action[:,1]*tf.cast(ydim,tf.float32))
	z = np.floor(action[:,2]*tf.cast(zdim,tf.float32))
	x = tf.cast(x,tf.int32)
	y = tf.cast(y,tf.int32)
	z = tf.cast(z,tf.int32)

	for i in range(len(action)):
		#print(x[i].numpy(),y[i].numpy(),z[i].numpy())
		if x[i] < xdim and y[i] < ydim and z[i] < zdim:
			if struct[0,x[i],y[i],z[i]] == 0:
				new_struct[0,x[i],y[i],z[i]] = 1
			else:
				new_struct[0,x[i],y[i],z[i]] = 0

	return new_struct

class EpisodeData:
	def __init__(self):
		self.observations = []
		self.actions = []
		self.rewards = []
		self.ts = [] # time step
		self.probs_old = []

def sample_episodes(obser, policy_network, num_episodes, maxlen, action_repeat=1):

	episodes = []
	og_struct = obser[:,:,:,:,0]
	structC = obser[:,:,:,:,1]
	structF = obser[:,:,:,:,2]
	stayoff = obser[:,:,:,:,3]

	xdim, ydim, zdim = tf.cast(tf.shape(og_struct[0]),tf.float32)
	#eng = start_engine()
	og_bend = eng.check_max_bend(convert_to_matlabint8(obser[0,:,:,:,0]), convert_to_matlabint8(structC[0]), convert_to_matlabint8(structF[0]))

	print("Generating episodes:")
	for i in range(num_episodes):
		episode = EpisodeData()
		observation = obser
		print("__________________________________")
		print("Episode: ", i)
		for t in range(maxlen):
			#observation = preprocess(observation)

			logits = policy_network.policy(observation)
			# remove num_samples dimension and batch dimension
			#action = tf.random.categorical(logits, 1)[0][0]
			#action = tf.random.categorical(logits, 150)[0]
			#action = tf.math.sigmoid(tf.cast(action,tf.float32))
			#action = tf.reshape(logits[0], [50,3])
			noise = tf.random.normal(shape = tf.shape(logits[0]), mean = 0.0, stddev = 0.05, dtype = tf.float32)
			action = logits[0] + noise
			action = tf.clip_by_value(action, 0, 1)
			#action = action/150
			pi_old = activations.softmax(logits)[0]
			
			episode.observations.append(observation[0])
			episode.ts.append(np.float32(t))
			episode.actions.append(action.numpy())
			episode.probs_old.append(pi_old.numpy())
			r = 0.0
			reward = 0.0 # accumulate reward accross actions
			#action = action_encoder.index2action(action).numpy()

			for _ in range(action_repeat):
				new_struct = flip_coord(action, observation[:,:,:,:,0])
				
				done = False
				if np.sum(new_struct) == 0:
					r = -100.0
					done = True

				elif np.sum(new_struct[0] - observation[0,:,:,:,0]) == 0:
					r = 0
					done = True

				else:
					try:
						#eng.clf(nargout=0)
						#eng.plotVg_safe(convert_to_matlabint8(new_struct[0]), 'edgeOff', 'col',collist, nargout=0)
						
						new_bend, comps = eng.check_max_bend(convert_to_matlabint8(new_struct[0]), 
							                                 convert_to_matlabint8(structC[0]), 
							                                 convert_to_matlabint8(structF[0]), nargout=2)
						
						if new_bend == 0 or np.isnan(new_bend):
							new_bend = -10.0
							#done = True

						vox_diff = np.abs(np.sum(og_struct.numpy()) - np.sum(new_struct))
						place_diff = np.abs(np.sum(og_struct.numpy() - new_struct))
						bend_diff = og_bend/new_bend
						
						#r = bend_diff + place_diff/5 - (vox_diff/10 + 10*(comps-1)) - r
						r = 1000*(bend_diff-1) - r
						print('{:16s} | {:.6f} | {:.6f}'.format('old | new bending:', og_bend, new_bend))
						print('{:16s} | {:3d} | {:3d}'.format('Voxels diff:', vox_diff, comps))
						if comps > 1:
							done = True
						
					except:
						comps = eng.check_components(convert_to_matlabint8(new_struct[0]), nargout=1)
						#r = - (vox_diff/10 + 10*(comps-1)) - r
						r = 0
						done = True
				reward = reward + r
				
				print('{:16s} {:.6f}'.format('Reward:', reward))

				if done:
					break

				

			episode.rewards.append(reward)
			observation = tf.stack([tf.convert_to_tensor(new_struct), structC, structF, stayoff], axis=4)
			if done:
				break

		episodes.append(episode)
		best_differences = new_struct[0] - og_struct[0].numpy()
		best_differences_minus = best_differences
		best_differences_positive = best_differences

		best_differences_minus[best_differences_minus > 0] = 0
		best_differences_minus[best_differences_minus < 0] = 1

		best_differences_positive[best_differences_positive < 0] = 0
		best_differences_positive[best_differences_positive > 0] = 1
		eng.clf(nargout=0)
		eng.plotVg_safe(convert_to_matlabint8(new_struct[0]), 'edgeOff', 'col',collist, 'alp', 0.05, nargout=0)
		eng.plotVg_safe(convert_to_matlabint8(best_differences_minus), 'edgeOff', 'col',collist3, 'alp', 0.5, nargout=0)
		eng.plotVg_safe(convert_to_matlabint8(best_differences_positive), 'edgeOff', 'col',collist2, 'alp', 0.5, nargout=0)
		
		eng.sun1(nargout=0)


	print(action)
	return episodes

def create_dataset(obser, policy_network, value_network, num_episodes, maxlen, action_repeat, gamma):

	episodes = sample_episodes(obser, policy_network, num_episodes, maxlen, action_repeat=action_repeat)

	dataset_size = 0
	for episode in episodes:
		# Could also get this when sampling episodes for efficiency
		# use predict?
		values = np.concatenate([value_network(np.expand_dims(o_t, 0), np.expand_dims(np.float32(maxlen)-t, 0)).numpy() for o_t, t in zip(episode.observations, episode.ts)])
		returns = calculate_returns(episode.rewards, gamma)
		advantages = returns - values

		episode.returns = returns
		episode.advantages = advantages
		dataset_size += len(episode.observations)

	slices = (
		tf.concat([e.observations for e in episodes], axis=0), # policy loss and value loss
		tf.concat([e.actions for e in episodes], axis=0), # policy loss
		tf.concat([e.advantages for e in episodes], axis=0), # policy loss
		tf.concat([e.probs_old for e in episodes], axis=0), # policy loss
		tf.concat([e.returns for e in episodes], axis=0), # value function loss
		tf.concat([e.ts for e in episodes], axis=0) # value function loss
	)
	dataset = tf.data.Dataset.from_tensor_slices(slices)
	dataset = dataset.shuffle(dataset_size)

	return dataset

def runstuff(train_dir, test_number, use_pre_struct=True, continue_train=True, show=False):
	# Construct model and measurements

	base_dir = os.path.join("reinforcement_model2", test_number)
	os.makedirs(base_dir, exist_ok=True)
	os.makedirs("results_structures2/", exist_ok=True)

	iterations = 500
	epoch_range = 20
	K = 3
	num_episodes = 12#2 #8
	maxlen_environment = 32
	action_repeat = 1
	maxlen = maxlen_environment // action_repeat # max number of actions
	batch_size = 1
	checkpoint_interval = 5
	eval_interval = 1
	eval_episodes = 1

	alpha_start = 1
	alpha_end = 0.0
	init_epsilon = 0.1
	init_lr = 0.5*10**-5
	optimizer = optimizers.Adam(init_lr)
	gamma = 0.99
	NUM_THREADS = 24
	c1 = 0.1 # value function loss weight
	c2 = 0.001 # entropy bonus weight

	#tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)
	#tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)

	trainAug = Sequential([
	layers.RandomFlip(mode="horizontal_and_vertical"),
	layers.RandomRotation(0.25)
	])

	#eng = start_engine()

	#structog, vGextC, vGextF, vGstayOff = eng.get_struct2(nargout=4)
	#structog, _, vGextC, vGextF, vGstayOff = eng.get_struct3(nargout=5)
	structog, _, vGextC, vGextF, vGstayOff = eng.get_struct4(nargout=5)
	#structog, vGextC, vGextF, vGstayOff = eng.get_struct5(nargout=4)
	og_maxbending = eng.check_max_bend(structog, vGextC, vGextF, nargout=1)

	struct = np.array(structog); structC = np.array(vGextC); structF = np.array(vGextF); structOff = np.array(vGstayOff)
	(xstruct,ystruct,zstruct) = struct.shape
	print(xstruct,ystruct,zstruct)
	struct = tf.convert_to_tensor(struct, dtype=tf.float32)
	structCten = tf.convert_to_tensor(structC, dtype=tf.float32)
	structFten = tf.convert_to_tensor(structF, dtype=tf.float32)
	structOfften = tf.convert_to_tensor(structOff, dtype=tf.float32)

	new_dataset = tf.data.Dataset.from_tensors((struct,structCten,structFten,structOfften))
	new_dataset = new_dataset.batch(batch_size)
	eng.clf(nargout=0)
	eng.plotVg_safe(structog, 'edgeOff', nargout=0)

	#eng.quit()

	#model = gen_model.ConvStructModel3D((x,y,z,4))

	# initialize policy and value network
	#action_encoder = ActionEncoder()
	feature_extractor = FeatureExtractor()
	policy_network = PolicyNetwork(feature_extractor)
	policy_network._set_inputs(np.zeros([1, xstruct,ystruct,zstruct, 4], dtype=np.float32))
	# Use to generate encoded actions (not just indices)
	agent = Agent(policy_network)
	agent._set_inputs(np.zeros([1, xstruct,ystruct,zstruct,4], dtype=np.float32))

	# possibly share parameters with policy-network
	value_network = ValueNetwork(feature_extractor, 100)

	mse = losses.MeanSquaredError()
	train_loss = metrics.Mean()
	val_loss = metrics.Mean()	
	mean_high = tf.Variable(-100000, dtype='float32', name='mean_high', trainable=False)
	#model.summary()
	#tf.keras.utils.plot_model(model, "3Dconv_model.png", show_shapes=True)

	#os.makedirs(os.path.dirname("training_reinforcement/" + test_number + "/"), exist_ok=True)
	checkpoint_path = base_dir + "/checkpoints/"
	#checkpoint_path = os.path.dirname(checkpoint_path)
	
	ckpt = tf.train.Checkpoint(
		policy_network=policy_network,
		value_network=value_network,
		mean_high=mean_high,
		optimizer=optimizer
		)

	ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
	if ckpt_manager.latest_checkpoint:
		print("Restored weights from {}".format(ckpt_manager.latest_checkpoint))
		ckpt.restore(ckpt_manager.latest_checkpoint)
	else:
		print("Initializing random weights.")

	start_iteration = 0
	if ckpt_manager.latest_checkpoint:
		start_iteration = int(ckpt_manager.latest_checkpoint.split('-')[-1])

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
		step_loss_vals = get_data("results_rl2/", test_number, "/step_loss.txt")
		step_diff_vals = get_data("results_rl2/", test_number, "/step_diff.txt")
		if step_loss_vals:
			step = step_loss_vals[-1][0]+1
		else:
			step = 0

		avg_loss_vals = get_data("results_rl2/", test_number, "/avg_train_loss.txt")
		avg_diff_vals = get_data("results_rl2/", test_number, "/avg_train_diff.txt")
		if avg_loss_vals:		
			saved_progress = avg_loss_vals[-1][0]+1
		else:
			saved_progress = 0
		
		step_val_loss_vals = get_data("results_rl2/", test_number, "/step_val_loss.txt")
		step_val_diff_vals = get_data("results_rl2/", test_number, "/step_val_diff.txt")
		if step_val_loss_vals:
			step = step_val_loss_vals[-1][0]+1
		else:
			step = 0
			
		avg_val_loss_vals = get_data("results_rl2/", test_number, "/avg_val_loss.txt")
		avg_val_diff_vals = get_data("results_rl2/", test_number, "/avg_val_diff.txt")
		"""
		try:
			latest = tf.train.latest_checkpoint(checkpoint_dir)
			model.load_weights(latest)

		except AttributeError:
			print("No previously saved weights")
		"""
	
	
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


	m_list, M_list = [], []
	median_list, mean_list = [], []
	iteration_list = []
	#struct, extC, extF, stayoff = new_dataset
	print("Starting from iteration: %d" % (start_iteration))
	for iteration in range(start_iteration, iterations):
		# linearly decay alpha, change epsilon and learning rate accordingly
		alpha = (alpha_start-alpha_end)*(iterations-iteration)/iterations+alpha_end
		epsilon = init_epsilon * alpha # clipping parameter

		optimizer.learning_rate.assign(init_lr*alpha) # set learning rate

		########################### Generate dataset ###########################
		start = time.time()
		for struct, structC, structF, stayoff in new_dataset.take(1):
			inpus = tf.stack([struct, structC, structF, stayoff], axis=4)
			xdim, ydim, zdim = tf.cast(tf.shape(struct[0]),tf.float32)
		dataset = create_dataset(inpus, policy_network, value_network,
								 num_episodes, maxlen,
								 action_repeat, gamma)

		print("Iteration: %d. Generated dataset in %f sec." %
			  (iteration, time.time() - start))
		print(alpha)
		dataset = dataset.batch(batch_size)
		
		for epoch in range(epoch_range):
			print(epoch, "/", epoch_range)
			start1 = time.time()
			# Trains model on structures with a truth structure created from
			# The direct stiffness method and shifted voxels
			for batch in dataset:
				obs, action, advantage, pi_old, value_target, t = batch
				#action = tf.expand_dims(action, -1)
				with tf.GradientTape() as tape:
					pi = activations.softmax(policy_network.policy(obs))
					v = value_network(obs, np.float32(maxlen)-t)
					
					#p_loss = policy_loss(pi, pi_old, advantage, epsilon)
					#v_loss = c1*value_loss(value_target, v)
					#e_loss = c2*entropy_loss(pi)
					loss = policy_loss(pi, pi_old, advantage, epsilon) \
						 + c1*value_loss(value_target, v) \
						 + c2*entropy_loss(pi)
				#print(p_loss, v_loss, e_loss)
				#loss = p_loss + v_loss + e_loss
				#print(tf.shape(pi))
				#print(tf.shape(action))
				#pi_a = tf.stack([pi, action], axis=1)[0]
				#pi_old_a = tf.stack([pi_old, action], axis=1)[0]
				#pi_a = tf.gather(pi, tf.cast(action, tf.int32))[0]
				#pi_old_a = tf.gather(pi_old, tf.cast(action, tf.int32))[0]
				#print(tf.shape(pi_a))
				#print(pi)
				#print(pi_a)
				#print(pi_old_a)

				trainable_variables = policy_network.trainable_variables + value_network.trainable_variables
				# Get unique list of variables, just adding lists may cause issues if shared variables
				trainable_variables = list({v.name : v for v in trainable_variables}.values())
				grads = tape.gradient(loss, trainable_variables)
				optimizer.apply_gradients(zip(grads, trainable_variables))
				#print(loss.numpy())
				# Update loss
				train_loss.update_state(loss)
			
			print("Time taken for one Epoch: ", time.time() - start1)
			"""
			if np.sum(out) != 0:
				try:
					new_maxbend = eng.check_max_bend(convert_to_matlabint8(out[0]), vGextC, vGextF, nargout=1)
					print("New vs old bending: ", new_maxbend, "/", og_maxbending)
				except:
					print("This one is singular :P")
			"""

			#if np.sum(out) != 0 and show:
			#	eng.clf(nargout=0)
			#	eng.plotVg_safe(convert_to_matlabint8(out[0]), 'edgeOff', 'col',collist, nargout=0)
		step += 1

		if step % checkpoint_interval == 0:
			print("Checkpointing model after %d iterations of training." % step)
			ckpt_manager.save(step)

		if step % eval_interval == 0:
			start = time.time()

			scores, best_episode, best_struct, best_differences = eval_policy(inpus, agent, maxlen_environment, 
				eval_episodes, action_repeat=action_repeat
			)
			best_differences_minus = np.abs(best_differences[best_differences < 0])
			best_differences_positive = np.abs(best_differences[best_differences > 0])
			m, M = np.min(scores), np.max(scores)
			median, mean = np.median(scores), np.mean(scores)

			iteration_list.append(iteration)
			m_list.append(m)
			M_list.append(M)
			median_list.append(median)
			mean_list.append(mean)

			plt.figure(1)
			plt.clf()
			plt.plot(iteration_list, m_list)
			plt.plot(iteration_list, M_list)
			plt.plot(iteration_list, median_list)
			plt.plot(iteration_list, mean_list)
			plt.grid()
			plt.legend(["Min", "Max", "Median", "Mean"])
			plt.title("Iteration Reward Progress")
			plt.xlabel("Iterations")
			plt.ylabel("Reward")
			plt.savefig("RL_progress2.png")

			print("Evaluated policy in %f sec. min, median, mean, max: (%g, %g, %g, %g)" %
				  (time.time() - start, m, median, mean, M))


			with open("results_structures2/" + test_number + "-" + str(step) + ".txt", "wb+") as fp:
				print("Writing best structure to results_structures/" + test_number + "-" + str(step) + ".txt")
				pickle.dump(best_struct, fp)
			"""
			with writer.as_default():
				tf.summary.scalar("return_min", m, step=step)
				tf.summary.scalar("return_max", M, step=step)
				tf.summary.scalar("return_mean", mean, step=step)
				tf.summary.scalar("return_median", median, step=step)
			"""
			eng.clf(nargout=0)
			eng.plotVg_safe(convert_to_matlabint8(best_struct[0]), 'edgeOff', 'col',collist, 'alp', 1, nargout=0)
			#eng.plotVg_safe(convert_to_matlabint8(best_differences_minus), 'edgeOff', 'col',collist3, 'alp', 0.8, nargout=0)
			#eng.plotVg_safe(convert_to_matlabint8(best_differences_positive), 'edgeOff', 'col',collist2, 'alp', 0.8, nargout=0)
			

			if mean > mean_high:
				print("New mean high! Old score: %g. New score: %g." %
					  (mean_high.numpy(), mean))
				mean_high.assign(mean)

				#tf.keras.models.save_model(agent, os.path.join(base_dir, "high_score_model"))
				#agent.save(os.path.join(base_dir, "high_score_model"))
	# Saving final model (in addition to highest scoring model already saved)
	# The model may be loaded with tf.keras.load_model(os.path.join(checkpoint_path, "agent"))
	#agent.save(os.path.join(checkpoint_path, "agent"))
		#if step % checkpoint_interval == 0:
		#	print("Checkpointing model after %d iterations of training." % step)
		#	ckpt_manager.save(step)

	print("Training completed.")

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
	runstuff(args.train_dir, args.test_number, use_pre_struct, continue_train)
