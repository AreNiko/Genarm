import numpy as np
import os
import matlab
import matlab.engine
import tensorflow as tf
import pickle
import random

def gen_data():
		#eng = matlab.engine.start_matlab()
		
	names = matlab.engine.find_matlab()
	print(names)
	if not names:
		eng = matlab.engine.start_matlab()
	else:
		eng = matlab.engine.connect_matlab(names[0])

	#path = "C:/Users/nikol/Documents/thesis/genarm/python_model/data/reinforce1"

	path = os.path.abspath(os.getcwd()) + "/data/reinforce1"

	struct1og, vgc, vGextC, vGextF, vGstayOff = eng.get_struct1(nargout=5)
	struct1 = np.array(struct1og)
	(x,y,z) = struct1.shape
	struct1C = np.array(vGextC)
	struct1F = np.array(vGextF)
	struct1Off = np.array(vGstayOff)
	datasize = 20
	structs = []
	rein = []
	for i in range(datasize):
		print(i)
		true_struct = np.int8(np.array(eng.reinforce_struct(matlab.int8(np.int8(struct1).tolist()), vGextC, 
													vGextF, vGstayOff, random.randint(50, 750))))
		
		structs.append(struct1)
		rein.append(true_struct)
		#dat[i] = np.array(struct1, np.array(true_struct))
		struct1 = true_struct

	structs = np.array(structs)
	rein = np.array(rein)
	dataset = tf.data.Dataset.from_tensor_slices((structs, rein))
	tf.data.experimental.save(dataset, path)
	#new_dataset = tf.data.experimental.load(path)

if __name__ == '__main__':
	gen_data()
