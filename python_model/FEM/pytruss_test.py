
import numpy as np
import matplotlib.pyplot as plt

import matlab
import matlab.engine

from vox2mesh18 import vox2mesh18
#from vox2mesh18_2 import vox2mesh18_2
from FEM_truss import FEM_truss
from vGextC2extC import vGextC2extC
from vGextF2extF import vGextF2extF

import time

def run_test():
	collist = matlab.double([0, 0.68, 0.7647])
	names = matlab.engine.find_matlab()

	if not names:
		eng = matlab.engine.start_matlab()
	else:
		eng = matlab.engine.connect_matlab(names[0])
	
	#structog, _, vGextC, vGextF, vGstayOff = eng.get_struct1(14,14,8,100, nargout=5)
	structog, vGextC, vGextF, vGstayOff = eng.get_struct2(nargout=4)
	struct = np.array(structog)
	(x,y,z) = struct.shape
	structC = np.array(vGextC)
	structF = np.array(vGextF)
	structOff = np.array(vGstayOff)

	eng.plotVg_safe(structog, 'edgeOff', 'col', collist, nargout=0)
	
	start1 = time.time()
	E,N,vGvokselNr = vox2mesh18(struct)

	radius = 0.003
	E[:,2] = np.pi*radius**2

	#Em,Nm,vGvokselNrm = eng.vox2mesh18(structog, nargout=3)
	#matlab_mesharr = np.array(Em)
	#print(E)
	#print(matlab_mesharr)
	
	extC = vGextC2extC(structC,struct)
	extF = vGextF2extF(structF,struct)
	extF = extF*np.array([0, 10, 0])
	extC[:,2] = 0
	
	sE, dN = FEM_truss(E,N,extF,extC)

	#dN[dN<1e-8] = 0
	stop1 = time.time() - start1
	
	start2 = time.time()
	sE_r, dN_r = eng.Struct_bend(structog, vGextC, vGextF, nargout=2)
	stop2 = time.time() - start2

	print(np.array(dN_r))
	print(np.max(np.abs(np.array(dN_r))))

	print(dN)
	print(np.max(np.abs(dN)))
	print(" Python implementasjon speed test: %f | Matlab implementasjon speed test: %f" % (stop1, stop2))		
	
if __name__ == '__main__':
	run_test()


