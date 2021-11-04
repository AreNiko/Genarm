
import numpy as np
import matplotlib.pyplot as plt

import matlab
import matlab.engine

import vox2mesh18
import FEM_truss

def run_test():
	collist = matlab.double([0, 0.68, 0.7647])
	names = matlab.engine.find_matlab()

	if not names:
		eng = matlab.engine.start_matlab()
	else:
		eng = matlab.engine.connect_matlab(names[0])
		
	structog, vgc, vGextC, vGextF, vGstayOff = eng.get_struct1(nargout=5)
	struct = np.array(structog)
	(x,y,z) = struct.shape
	structC = np.array(vGextC)
	structF = np.array(vGextF)
	structOff = np.array(vGstayOff)

	eng.plotVg_safe(structog, 'edgeOff', nargout=0)

	python_mesh = vox2mesh18(struct)
	matlab_mesh = eng.vox2mesh18(structog)

	matlab_mesharr = np.array(matlab_mesh)
	comparison = python_mesh == matlab_mesharr
	print(comparison.all)
		
	
if __name__ == '__main__':
	run_test()


