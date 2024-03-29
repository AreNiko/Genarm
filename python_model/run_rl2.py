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
		future = matlab.engine.start_matlab(background=True)
		eng = future.result()
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

	#og_bend = eng.check_max_bend(convert_to_matlabint8(obser[0,:,:,:,0]), convert_to_matlabint8(structC[0]), convert_to_matlabint8(structF[0]))
	og_bend, new_disp, _ = eng.check_max_stress(convert_to_matlabint8(obser[0,:,:,:,0]), convert_to_matlabint8(structC[0]), convert_to_matlabint8(structF[0]), nargout=3)
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
			action = agent(observation)
			# remove num_samples dimension and batch dimension
			#action = tf.random.categorical(logits[0], 1)

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
						#new_bend, comps = eng.check_max_bend(convert_to_matlabint8(new_struct[0]),
						#	                                 convert_to_matlabint8(structC[0]),
						#	                                 convert_to_matlabint8(structF[0]), nargout=2)

						new_bend, new_disp, comps = eng.check_max_stress(convert_to_matlabint8(new_struct[0]),
							                                 convert_to_matlabint8(structC[0]),
							                                 convert_to_matlabint8(structF[0]), nargout=3)
						if new_bend == 0 or np.isnan(new_bend):
							new_bend = og_bend
							#done = True

						vox_diff = np.abs(np.sum(og_struct.numpy()) - np.sum(new_struct))
						place_diff = np.abs(np.sum(og_struct.numpy() - new_struct))

						bend_diff = og_bend/new_bend

						#r = bend_diff + place_diff/5 - (vox_diff/10 + 10*(comps-1))
						#r = 10*(bend_diff - 1) - (vox_diff + (comps-1))/100
						#r = 2*(bend_diff - 1)
						#reward = 1000*(og_bend - new_bend) - (vox_diff + 5*(comps-1))/100
						reward = (og_bend - new_bend)/1000 + (1 - (comps-1))/10
						print('| {:14s} | {:14f} |'.format('Old Stress:', og_bend))
						print('| {:14s} | {:14f} |'.format('New STress:', new_bend))
						print('| {:14s} | {:14d} |'.format('Voxels diff:', int(vox_diff)))
						print('| {:14s} | {:14d} |'.format('Nr components:', int(comps)))

						if comps > 1:
							done = True

					except:
						comps = eng.check_components(convert_to_matlabint8(new_struct[0]), nargout=1)
						vox_diff = np.abs(np.sum(og_struct.numpy()) - np.sum(new_struct))
						#reward = - (vox_diff/10 + (comps-1))
						reward = (1 - (comps-1))/10
						#reward = 0
						if comps > 1:
							done = True

				if best_reward < reward or best_reward is None:
					best_struct = new_struct
					best_differences = new_struct[0] - og_struct[0].numpy()

				print('| {:14s} | {:14f} |\n'.format('Reward:', reward))
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

	return scores, best_episode, best_struct

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
	action = action.numpy().T
	new_struct = struct.numpy()
	batch, xdim, ydim, zdim = tf.shape(struct)
	#action = tf.cast(action,tf.int32)

	for i in range(len(action)):
		if action[i,0] >= xdim:
			action[i,0] = xdim-1
		if action[i,1] >= ydim:
			action[i,1] = ydim-1
		if action[i,2] >= zdim:
			action[i,2] = zdim-1

		#print(x[i].numpy(),y[i].numpy(),z[i].numpy())
		if action[i,0] < xdim and action[i,1] < ydim and action[i,2] < zdim:
			if struct[0,action[i,0],action[i,1],action[i,2]] == 0:
				new_struct[0,action[i,0],action[i,1],action[i,2]] = 1
			else:
				new_struct[0,action[i,0],action[i,1],action[i,2]] = 0

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
	original_stress, original_bend, og_comps = eng.check_max_stress(convert_to_matlabint8(obser[0,:,:,:,0]), convert_to_matlabint8(structC[0]), convert_to_matlabint8(structF[0]), nargout=3)
	print(original_stress)
	print("Generating episodes:")
	for i in range(num_episodes):
		episode = EpisodeData()
		observation = obser
		og_stress = original_stress
		og_bend = original_bend
		print("__________________________________")
		print("Episode: ", i)
		for t in range(maxlen):
			#observation = preprocess(observation)

			logits = activations.softmax(policy_network.policy(observation))

			# remove num_samples dimension and batch dimension
			#action = tf.random.categorical(logits, 1)[0][0]

			action = tf.random.categorical(logits[0], 1)

			#actiony = tf.random.categorical(logitsy[0], 1)
			#actionz = tf.random.categorical(logitsz[0], 1)
			#action = tf.stack([actionx[:,0], actiony[:,0], actionz[:,0]], axis=0)
			#print(action)

			#action = tf.math.sigmoid(tf.cast(action,tf.float32))
			#action = tf.reshape(logits[0], [50,3])
			#print(logits)
			"""
			noise = tf.random.normal(shape = tf.shape(logits[0]), mean = 0.0, stddev = 0.05, dtype = tf.float32)
			action = logits[0] + noise
			#action = tf.clip_by_value(action, 0, 1)
			action = action.numpy()

			action[:,0] = np.floor(action[:,0]*tf.cast(xdim,tf.float32))
			action[:,0] = tf.clip_by_value(action[:,0], 0, xdim)
			action[:,1] = np.floor(action[:,1]*tf.cast(ydim,tf.float32))
			action[:,1] = tf.clip_by_value(action[:,1], 0, ydim)
			action[:,2] = np.floor(action[:,2]*tf.cast(zdim,tf.float32))
			action[:,2] = tf.clip_by_value(action[:,2], 0, zdim)
			"""
			#action = action/150
			#pi_old = tf.concat([logitsx[0], logitsy[0], logitsz[0]], 1)
			pi_old = activations.softmax(logits)[0]
			episode.observations.append(observation[0])
			episode.ts.append(np.float32(t))
			episode.actions.append(action)
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

						new_stress, new_bend, comps = eng.check_max_stress(convert_to_matlabint8(new_struct[0]),
							                                 convert_to_matlabint8(structC[0]),
							                                 convert_to_matlabint8(structF[0]), nargout=3)

						if new_bend == 0 or np.isnan(new_bend):
							new_bend = og_bend
							#done = True

						vox_diff = np.abs(np.sum(og_struct.numpy()) - np.sum(new_struct))
						place_diff = np.abs(np.sum(og_struct.numpy() - new_struct))

						bend_diff = og_bend/new_bend

						#r = bend_diff + place_diff/5 - (vox_diff/10 + 10*(comps-1))
						#r = 10*(bend_diff - 1) - (vox_diff + (comps-1))/100
						#r = 2*(bend_diff - 1)
						r = 1000*(og_bend - new_bend) - (vox_diff + comps-1)/100
						#r = 1000*(og_bend - new_bend)
						#r = (og_stress - new_stress)/1000 + (1 - (comps-1))/10
						print('| {:14s} | {:14f} |'.format('Old bending:', og_bend))
						print('| {:14s} | {:14f} |'.format('New bending:', new_bend))
						print('| {:14s} | {:14f} |'.format('Bending diff:', bend_diff))
						print('| {:14s} | {:14d} |'.format('Voxels diff:', int(vox_diff)))
						print('| {:14s} | {:14d} |'.format('Nr components:', int(comps)))
						if new_bend != 1.0:
							og_bend = new_bend

						if comps > 1:
							done = True

					except:
						comps = eng.check_components(convert_to_matlabint8(new_struct[0]), nargout=1)
						#r = - (vox_diff + (comps-1))/1
						r = (-(comps-1))/10
						#r = 0
						if comps > 1:
							done = True
				reward = reward + r

				print('| {:14s} | {:14f} |\n'.format('Reward:', reward))

				if done:
					break



			episode.rewards.append(reward)
			observation = tf.stack([tf.convert_to_tensor(new_struct), structC, structF, stayoff], axis=4)
			if done:
				break

		episodes.append(episode)
		best_differences = np.abs(new_struct[0] - og_struct[0].numpy())
		"""
		best_differences_minus = best_differences
		best_differences_positive = best_differences

		best_differences_minus[best_differences_minus > 0] = 0
		best_differences_minus[best_differences_minus < 0] = 1

		best_differences_positive[best_differences_positive < 0] = 0
		best_differences_positive[best_differences_positive > 0] = 1
		eng.clf(nargout=0)
		#eng.plotVg_safe(convert_to_matlabint8(new_struct[0]), 'edgeOff', 'col',collist, 'alp', 0.05, nargout=0)
		eng.plotVg_safe(convert_to_matlabint8(best_differences_minus), 'edgeOff', 'col',collist3, 'alp', 0.5, nargout=0)
		eng.plotVg_safe(convert_to_matlabint8(best_differences_positive), 'edgeOff', 'col',collist2, 'alp', 0.5, nargout=0)
		"""
		eng.clf(nargout=0)
		eng.plotVg_safe(convert_to_matlabint8(new_struct[0]), 'edgeOff', 'col',collist, 'alp', 0.05, nargout=0)
		eng.plotVg_safe(convert_to_matlabint8(best_differences), 'edgeOff', 'col',collist3, 'alp', 0.5, nargout=0)
		eng.sun1(nargout=0)

	return episodes

def create_dataset(obser, policy_network, value_network, num_episodes, maxlen, action_repeat, gamma, iteration, test_number):

	episodes = sample_episodes(obser, policy_network, num_episodes, maxlen, action_repeat=action_repeat)

	dataset_size = 0
	ep_number = 0
	episode_legend = []
	episode_avg = 0
	plt.figure(1)
	plt.clf()
	episode_max = -100000000000
	episode_min = 100000000000
	all_episode_reward = []
	for episode in episodes:
		# Could also get this when sampling episodes for efficiency
		# use predict?
		values = np.concatenate([value_network(np.expand_dims(o_t, 0), np.expand_dims(np.float32(maxlen)-t, 0)).numpy() for o_t, t in zip(episode.observations, episode.ts)])
		returns = calculate_returns(episode.rewards, gamma)
		advantages = returns - values

		episode.returns = returns
		episode.advantages = advantages
		dataset_size += len(episode.observations)

		if episode_max < np.sum(episode.rewards):
			episode_max = np.sum(episode.rewards)
		if episode_min > np.sum(episode.rewards):
			episode_min = np.sum(episode.rewards)

		all_episode_reward.append(episode.rewards)
		episode_avg += np.sum(episode.rewards)
		plt.plot(episode.ts, episode.rewards)
		episode_legend.append("Episode " + str(ep_number))
		ep_number += 1
	plt.grid()
	plt.legend(episode_legend, bbox_to_anchor=(1.04,1), loc="upper left")
	plt.title("Training Episode Reward Progress")
	plt.xlabel("Time Step")
	plt.ylabel("Reward")
	plt.savefig("rl2_plots/"+ test_number + "/iteration_" + str(iteration) + ".png")

	episode_avg /= num_episodes
	with open("rl2_data/" + test_number + "/" + "Episode-stats_" + str(iteration) + ".txt", "wb+") as fp:
		pickle.dump([iteration, episode_max, episode_min, episode_avg, all_episode_reward], fp)

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
	os.makedirs("rl2_plots/" + test_number + "/", exist_ok=True)
	os.makedirs("rl2_data/" + test_number + "/", exist_ok=True)


	iterations = 2000
	epoch_range = 3
	num_episodes = 32#2 #8
	maxlen_environment = 16
	action_repeat = 1
	maxlen = maxlen_environment // action_repeat # max number of actions
	batch_size = 1
	checkpoint_interval = 1
	eval_interval = 1
	eval_episodes = 12

	alpha_start = 1
	alpha_end = 0.0
	init_epsilon = 0.2
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

	structog, vGextC, vGextF, vGstayOff = eng.get_struct2(nargout=4)
	#structog, _, vGextC, vGextF, vGstayOff = eng.get_struct3(nargout=5)
	#structog, _, vGextC, vGextF, vGstayOff = eng.get_struct4(nargout=5)
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
	policy_network.set_coords([1, xstruct,ystruct,zstruct, 4])
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
	writer = tf.summary.create_file_writer(base_dir  + "summary3/")
	kl_divergence = losses.KLDivergence(reduction=losses.Reduction.NONE)
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
								 action_repeat, gamma, iteration, test_number)

		print("Iteration: %d. Generated dataset in %f sec." %
			  (iteration, time.time() - start))
		print(alpha)
		dataset = dataset.batch(batch_size)

		for epoch in range(epoch_range):
			print(epoch, "/", epoch_range)
			start1 = time.time()

			if epoch == 0:
				episode_legend = []
				t_values = []
				reward_values = []
				ep_number = 0
			# Trains model on structures with a truth structure created from
			# The direct stiffness method and shifted voxels
			for batch in dataset:

				#action = tf.expand_dims(action, -1)
				with tf.GradientTape() as tape:
					#pi = activations.softmax(policy_network.policy(obs))
					obs, action, advantage, pi_old, value_target, t = batch
					logs = policy_network.policy(obs)
					action = tf.expand_dims(action, -1)
					pi = activations.softmax(logs)
					#pi = tf.concat([logitsx[0], logitsy[0], logitsz[0]], 1)
					#pi = activations.softmax(pi)
					#print(pi[0])
					#print(action[0])
					pi_a = tf.squeeze(tf.gather(pi[0], action[0], batch_dims=1), -1)
					pi_old_a = tf.squeeze(tf.gather(pi_old[0], action[0], batch_dims=1), -1)
					v = value_network(obs, np.float32(maxlen)-t)
					#print(pi_a)
					#pi_a = tf.tensor([pi[0,action[0]]
					#pi_a = tf.concat([pi, tf.cast(action, tf.float32)], axis=-1)
					#pi_old_a = tf.concat([pi_old, tf.cast(action, tf.float32)], axis=-1)
					#pi_a = tf.stack([pi, action], axis=2)
					#pi_old_a = tf.stack([pi_old, action], axis=2)

					loss = policy_loss(pi_a, pi_old_a, advantage, epsilon) \
						 + c1*value_loss(value_target, v) \
						 + c2*entropy_loss(pi)

				trainable_variables = policy_network.trainable_variables + value_network.trainable_variables
				# Get unique list of variables, just adding lists may cause issues if shared variables
				trainable_variables = list({v.name : v for v in trainable_variables}.values())
				grads = tape.gradient(loss, trainable_variables)
				optimizer.apply_gradients(zip(grads, trainable_variables))
				#print(loss.numpy())
				# Update loss
				train_loss.update_state(loss)


			print("Time taken for one epoch: ", time.time() - start1)
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

		ratios, entropies, entropies_old, actions, advantages, kl_divs, diff, diff_lb = [], [], [], [], [], [], [], []
		for batch in dataset:
			obs, action, advantage, pi_old, value_target, t = batch
			action = tf.expand_dims(action, -1)[0]
			logits = policy_network.policy(obs)
			pi = activations.softmax(logits)[0]
			pi_old = pi_old[0]
			v = value_network(obs, np.float32(maxlen)-t)
			pi_a = tf.squeeze(tf.gather(pi, action, batch_dims=1), -1)
			pi_old_a = tf.squeeze(tf.gather(pi_old, action, batch_dims=1), -1)
			ratio = pi_a / pi_old_a

			kl_divs.append(kl_divergence(pi_old, pi))
			ratios.append(ratio)
			advantages.append(advantage)
			entropies.append(entropy(pi))
			entropies_old.append(entropy(pi_old))
			actions.append(tf.one_hot(tf.squeeze(action, -1), 1000))
			diff.append(estimate_improvement(pi_a, pi_old_a, advantage, t, gamma))
			diff_lb.append(estimate_improvement_lb(pi_a, pi_old_a, advantage, t, gamma, epsilon))

		actions = tf.concat(actions, axis=0)
		action_frequencies = tf.reduce_sum(actions, axis=0) / tf.reduce_sum(actions)
		action_entropy = entropy(action_frequencies)
		ratios = tf.concat(ratios, axis=0)
		entropies = tf.concat(entropies, axis=0)
		entropies_old = tf.concat(entropies_old, axis=0)
		advantages = tf.concat(advantages, axis=0)
		diff = tf.concat(diff, axis=0)
		diff_lb = tf.concat(diff_lb, axis=0)
		kl_divs = tf.concat(kl_divs, axis=0)
		with writer.as_default():
			tf.summary.histogram("advantages", advantages, step=step)
			tf.summary.histogram("sample_improvements", diff, step=step)
			tf.summary.histogram("sample_improvements_lb", diff_lb, step=step)
			tf.summary.scalar("estimated_improvement",
				tf.reduce_sum(diff)/num_episodes, step=step)
			tf.summary.scalar("estimated_improvement_lb",
				tf.reduce_sum(diff_lb)/num_episodes, step=step)
			tf.summary.scalar("mean_advantage", tf.reduce_mean(advantages), step=step)
			tf.summary.histogram("entropy", entropies, step=step)
			tf.summary.histogram("prob_ratios", ratios, step=step)
			tf.summary.scalar("mean_entropy", tf.reduce_mean(entropies), step=step)
			tf.summary.scalar("mean_entropy_old", tf.reduce_mean(entropies_old), step=step)
			tf.summary.histogram("kl_divergence", kl_divs, step=step)
			tf.summary.scalar("mean_kl_divergence", tf.reduce_mean(kl_divs), step=step)
			tf.summary.scalar("kl_divergence_max", tf.reduce_max(kl_divs), step=step)
			#tf.summary.scalar("action_entropy", action_entropy, step=step)
			#tf.summary.scalar("action_freq_left", action_frequencies[0], step=step)
			#tf.summary.scalar("action_freq_straight", action_frequencies[1], step=step)
			#tf.summary.scalar("action_freq_right", action_frequencies[2], step=step)
			#tf.summary.scalar("action_freq_gas", action_frequencies[3], step=step)
			#tf.summary.scalar("action_freq_break", action_frequencies[4], step=step)

		if step % checkpoint_interval == 0:
			print("Checkpointing model after %d iterations of training." % step)
			ckpt_manager.save()

		if step % eval_interval == 0:
			start = time.time()

			scores, best_episode, best_struct = eval_policy(inpus, agent, maxlen_environment,
				eval_episodes, action_repeat=action_repeat
			)
			#best_differences_minus = np.abs(best_differences[best_differences < 0])
			#best_differences_positive = np.abs(best_differences[best_differences > 0])
			m, M = np.min(scores), np.max(scores)
			median, mean = np.median(scores), np.mean(scores)

			iteration_list.append(iteration)
			m_list.append(m)
			M_list.append(M)
			median_list.append(median)
			mean_list.append(mean)

			plt.figure(2)
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
