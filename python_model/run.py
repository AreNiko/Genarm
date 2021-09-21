import glob
import os
import argparse
import time
from functools import partial
import matlab.engine

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, metrics, optimizers

import model



def generate_random_struct():
	struct = tf.random.uniform(shape=(3), minval=0, maxval=1, dtype=tf.int32)
	return struct

def DSM(x):
	
	return y

def runstuff():
	struct1 = generate_random_struct()
	eng = matlab.engine.start_matlab()
	def loss_func(x):
		# Apply direct stiffness method as loss function
		(E, N,_) = eng.vox2mesh18(x);
		(sE, dN) = eng.FEM_truss(N,E, extF,extC)

	return loss


if __name__ == '__main__':
	runstuff()
