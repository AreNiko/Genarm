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

	path = os.path.abspath(os.getcwd()) + "/data/reinforce_all/006_sf" # _sf = start&finish, _step = start-finish

	
	datasize = 500
	structs = []
	rein = []
	struct1C_list = []
	struct1F_list = []
	struct1off_list = []
	collist = matlab.double([0, 0.68, 0.7647])
	for i in range(500):
		print(i)
		radii = np.random.randint(10, 14)
		wrist_rad = radii - np.random.randint(1, 3)
		struct1og, vGc, vGextC, vGextF, vGstayOff = eng.get_struct1(radii,radii,wrist_rad,np.random.randint(40,100), nargout=5)
		struct1 = np.array(struct1og, dtype=np.int8)
		(x,y,z) = struct1.shape
		struct1C = np.array(vGextC, dtype=np.int8)
		struct1F = np.array(vGextF, dtype=np.int8)
		struct1Off = np.array(vGstayOff, dtype=np.int8)
		for j in range(100):
			true_struct = np.array(eng.reinforce_struct(matlab.int8(struct1.tolist()), vGextC, 
														vGextF, vGstayOff, 50), dtype=np.int8)
			"""
			eng.figure(matlab.double(1), nargout=1)
			eng.clf(nargout=0)
			eng.plotVg_safe(matlab.int8(np.int8(np.ceil(true_struct)).tolist()), 'edgeOff', nargout=0)
			eng.plotVg_safe(matlab.int8(np.int8(np.ceil(vGextC)).tolist()), 'edgeOff', 'col',matlab.double([0.6, 0.6, 0.8]), nargout=0)
			eng.plotVg_safe(matlab.int8(np.int8(np.ceil(vGextF)).tolist()), 'edgeOff', 'col',matlab.double([0.5, 0.5, 0.5]), nargout=0)
			eng.plotVg_safe(matlab.int8(np.int8(np.ceil(vGstayOff)).tolist()), 'edgeOff', 'col',matlab.double([0.9, 0.9, 0.5]), nargout=0)
			"""
		structs.append(struct1)
		rein.append(true_struct)
		struct1C_list.append(struct1C)
		struct1F_list.append(struct1F)
		struct1off_list.append(struct1Off)
		#dat[i] = np.array(struct1, np.array(true_struct))
		struct1 = true_struct

	structs = np.array(structs, dtype=np.int8)
	rein = np.array(rein, dtype=np.int8)
	struct1C_list = np.array(struct1C_list, dtype=np.int8)
	struct1F_list = np.array(struct1F_list, dtype=np.int8)
	struct1off_list = np.array(struct1off_list, dtype=np.int8)
	dataset = tf.data.Dataset.from_tensor_slices((structs, struct1C_list, struct1F_list, struct1off_list, rein))
	tf.data.experimental.save(dataset, path)
	#new_dataset = tf.data.experimental.load(path)

if __name__ == '__main__':
	gen_data()
