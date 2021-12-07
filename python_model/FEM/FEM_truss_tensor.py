import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse.linalg
import tensorflow as tf
#from pysparse import pysparse

#@tf.function
def FEM_truss(E,N,extF,extC):

	verbose = 0
	EModulus = 2e9
	#readVgrarginInput(); % se nederst
	A = E[:,2] # Areal pï¿½ stag tverrsnitt, user spesifisert

	#extC = np.reshape(extC, 1) # TODO
	#extF = np.reshape(extF, 1) # TODO

	extC = extC.flatten()
	extF = extF.flatten()

	numberOfEdges = len(E)
	numberOfNodes = len(N)
	kDof = 3*numberOfNodes

	#dim = numberOfEdges*9*2 + numberOfNodes*3 + (numberOfNodes*3-1)*2 # antall 3X3 clash + 3 diagonal linjer, vil bli redusert
	dim = 36*numberOfEdges
	I = np.zeros(dim) # X, ned
	J = np.zeros(dim) # y, bort
	X = np.zeros(dim)

	IJindexTeller = 0
	for e in range(numberOfEdges): # runner alle edger/element
		indice = E[e,:] # 2 ende nodenummer [n1 n2] pï¿½ edge
		# genererer indekser i K for 6x6 matrise basert pï¿½ edge endepunkt x1 y1 z1  x2 y2 z2, n1 og n2 er ikke nï¿½dv naboer i K
		edgeDof=[3*indice[0], 3*indice[0]+1, 3*indice[0]+2, 3*indice[1], 3*indice[1]+1, 3*indice[1]+2] # [3a-2 3a+1 3a  3b-2 3b-1 3b]

		x1=N[int(indice[0]),0] # venstre node nr i edge sin x verdi
		y1=N[int(indice[0]),1] # venstre node nr i edge sin y verdi
		z1=N[int(indice[0]),2] # venstre node nr i edge sin z verdi
		x2=N[int(indice[1]),0] # hï¿½yre node nr i edge sin x verdi
		y2=N[int(indice[1]),1] # hï¿½yre node nr i edge sin y verdi
		z2=N[int(indice[1]),2] # hï¿½yre node nr i edge sin z verdi
		L = np.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1)) # lengde pï¿½ edge i 3D
		
		CXx = (x2-x1)/L # lengde pï¿½ edge i x retning / total lengde
		CYx = (y2-y1)/L # lengde pï¿½ edge i y retning / total lengde
		CZx = (z2-z1)/L # lengde pï¿½ edge i z retning / total lengde
		
		T = EModulus*A[e]/L*np.array([[CXx*CXx, CXx*CYx, CXx*CZx],
			 [CYx*CXx, CYx*CYx, CYx*CZx], 
			 [CZx*CXx, CZx*CYx, CZx*CZx]]) # [3,3] matrise

		#IJindexTeller = IJindexTeller+1
		#1
		I[ IJindexTeller ] = edgeDof[0] # x index
		J[ IJindexTeller ] = edgeDof[0] # y index
		X[ IJindexTeller ] = T[0,0] # verdi
		
		IJindexTeller = IJindexTeller+1
		# celle x1 bort, x1 ned
		#2
		I[ IJindexTeller ] = edgeDof[0] # x index
		J[ IJindexTeller ] = edgeDof[1] # y index
		X[ IJindexTeller ] = T[0,1] # verdi

		IJindexTeller = IJindexTeller+1
		# celle y1 bort, x1 ned
		#3
		I[ IJindexTeller ] = edgeDof[0] # x index
		J[ IJindexTeller ] = edgeDof[2] # y index
		X[ IJindexTeller ] = T[0,2] # verdi
		
		IJindexTeller = IJindexTeller+1
		# celle x1 bort, x1 ned
		#4
		I[ IJindexTeller ] = edgeDof[1] # x index
		J[ IJindexTeller ] = edgeDof[0] # y index
		X[ IJindexTeller ] = T[1,0] # verdi

		IJindexTeller = IJindexTeller+1
		# celle y1 bort, x1 ned
		#5
		I[ IJindexTeller ] = edgeDof[1] # x index
		J[ IJindexTeller ] = edgeDof[1] # y index
		X[ IJindexTeller ] = T[1,1] # verdi

		IJindexTeller = IJindexTeller+1
		#6
		I[ IJindexTeller ] = edgeDof[1] # x index
		J[ IJindexTeller ] = edgeDof[2] # y index
		X[ IJindexTeller ] = T[1,2] # verdi

		IJindexTeller = IJindexTeller+1
		#7
		I[ IJindexTeller ] = edgeDof[2] # x index
		J[ IJindexTeller ] = edgeDof[0] # y index
		X[ IJindexTeller ] = T[2,0] # verdi

		IJindexTeller = IJindexTeller+1
		#8
		I[ IJindexTeller ] = edgeDof[2] # x index
		J[ IJindexTeller ] = edgeDof[1] # y index
		X[ IJindexTeller ] = T[2,1] # verdi

		IJindexTeller = IJindexTeller+1
		#9
		I[ IJindexTeller ] = edgeDof[2] # x index
		J[ IJindexTeller ] = edgeDof[2] # y index
		X[ IJindexTeller ] = T[2,2] # verdi

		# oppe hï¿½yre bolk
		IJindexTeller = IJindexTeller+1
		# celle x1 bort, x1 ned
		#10
		I[ IJindexTeller ] = edgeDof[3] # x index
		J[ IJindexTeller ] = edgeDof[3] # y index
		X[ IJindexTeller ] = T[0,0] # verdi

		IJindexTeller = IJindexTeller+1
		# celle x1 bort, x1 ned
		#11
		I[ IJindexTeller ] = edgeDof[3] # x index
		J[ IJindexTeller ] = edgeDof[4] # y index
		X[ IJindexTeller ] = T[0,1] # verdi

		IJindexTeller = IJindexTeller+1
		# celle x1 bort, x1 ned
		#12
		I[ IJindexTeller ] = edgeDof[3] # x index
		J[ IJindexTeller ] = edgeDof[5] # y index
		X[ IJindexTeller ] = T[0,2] # verdi

		IJindexTeller = IJindexTeller+1
		# celle x1 bort, x1 ned
		#13
		I[ IJindexTeller ] = edgeDof[4] # x index
		J[ IJindexTeller ] = edgeDof[3] # y index
		X[ IJindexTeller ] = T[1,0] # verdi

		IJindexTeller = IJindexTeller+1
		# celle y1 bort, x1 ned
		#14
		I[ IJindexTeller ] = edgeDof[4] # x index
		J[ IJindexTeller ] = edgeDof[4] # y index
		X[ IJindexTeller ] = T[1,1] # verdi

		IJindexTeller = IJindexTeller+1
		# celle y1 bort, x1 ned
		#15
		I[ IJindexTeller ] = edgeDof[4] # x index
		J[ IJindexTeller ] = edgeDof[5] # y index
		X[ IJindexTeller ] = T[1,2] # verdi


		IJindexTeller = IJindexTeller+1
		# celle x1 bort, x1 ned
		#16
		I[ IJindexTeller ] = edgeDof[5] # x index
		J[ IJindexTeller ] = edgeDof[3] # y index
		X[ IJindexTeller ] = T[2,0] # verdi

		IJindexTeller = IJindexTeller+1
		# celle x1 bort, x1 ned
		#17
		I[ IJindexTeller ] = edgeDof[5] # x index
		J[ IJindexTeller ] = edgeDof[4] # y index
		X[ IJindexTeller ] = T[2,1] # verdi

		IJindexTeller = IJindexTeller+1
		# celle x1 bort, x1 ned
		#18
		I[ IJindexTeller ] = edgeDof[5] # x index
		J[ IJindexTeller ] = edgeDof[5] # y index
		X[ IJindexTeller ] = T[2,2] # verdi

		IJindexTeller = IJindexTeller+1
		# celle x1 bort, x1 ned
		#19
		I[ IJindexTeller ] = edgeDof[0] # x index
		J[ IJindexTeller ] = edgeDof[3] # y index
		X[ IJindexTeller ] = -T[0,0] # verdi

		IJindexTeller = IJindexTeller+1
		# celle x1 bort, x1 ned
		#20
		I[ IJindexTeller ] = edgeDof[0] # x index
		J[ IJindexTeller ] = edgeDof[4] # y index
		X[ IJindexTeller ] = -T[0,1] # verdi

		IJindexTeller = IJindexTeller+1
		# celle x1 bort, x1 ned
		#21
		I[ IJindexTeller ] = edgeDof[0] # x index
		J[ IJindexTeller ] = edgeDof[5] # y index
		X[ IJindexTeller ] = -T[0,2] # verdi

		IJindexTeller = IJindexTeller+1
		# celle x1 bort, x1 ned
		#22
		I[ IJindexTeller ] = edgeDof[1] # x index
		J[ IJindexTeller ] = edgeDof[3] # y index
		X[ IJindexTeller ] = -T[1,0] # verdi

		IJindexTeller = IJindexTeller+1
		# celle x1 bort, x1 ned
		#23
		I[ IJindexTeller ] = edgeDof[1] # x index
		J[ IJindexTeller ] = edgeDof[4] # y index
		X[ IJindexTeller ] = -T[1,1] # verdi

		IJindexTeller = IJindexTeller+1
		# celle x1 bort, x1 ned
		#24
		I[ IJindexTeller ] = edgeDof[1] # x index
		J[ IJindexTeller ] = edgeDof[5] # y index
		X[ IJindexTeller ] = -T[1,2] # verdi

		IJindexTeller = IJindexTeller+1
		# celle x1 bort, x1 ned
		#25
		I[ IJindexTeller ] = edgeDof[2] # x index
		J[ IJindexTeller ] = edgeDof[3] # y index
		X[ IJindexTeller ] = -T[2,0] # verdi

		IJindexTeller = IJindexTeller+1
		# celle x1 bort, x1 ned
		#26
		I[ IJindexTeller ] = edgeDof[2] # x index
		J[ IJindexTeller ] = edgeDof[4] # y index
		X[ IJindexTeller ] = -T[2,1] # verdi

		IJindexTeller = IJindexTeller+1
		# celle x1 bort, x1 ned
		#27
		I[ IJindexTeller ] = edgeDof[2] # x index
		J[ IJindexTeller ] = edgeDof[5] # y index
		X[ IJindexTeller ] = -T[2,2] # verdi

		IJindexTeller = IJindexTeller+1
		# celle x1 bort, x1 ned
		#28
		I[ IJindexTeller ] = edgeDof[3] # x index
		J[ IJindexTeller ] = edgeDof[0] # y index
		X[ IJindexTeller ] = -T[0,0] # verdi

		IJindexTeller = IJindexTeller+1
		# celle x1 bort, x1 ned
		#29
		I[ IJindexTeller ] = edgeDof[3] # x index
		J[ IJindexTeller ] = edgeDof[1] # y index
		X[ IJindexTeller ] = -T[0,1] # verdi

		IJindexTeller = IJindexTeller+1
		# celle x1 bort, x1 ned
		#30
		I[ IJindexTeller ] = edgeDof[3] # x index
		J[ IJindexTeller ] = edgeDof[2] # y index
		X[ IJindexTeller ] = -T[0,2] # verdi

		IJindexTeller = IJindexTeller+1
		# celle x1 bort, x1 ned
		#31
		I[ IJindexTeller ] = edgeDof[4] # x index
		J[ IJindexTeller ] = edgeDof[0] # y index
		X[ IJindexTeller ] = -T[1,0] # verdi

		IJindexTeller = IJindexTeller+1
		# celle y1 bort, x1 ned
		#32
		I[ IJindexTeller ] = edgeDof[4] # x index
		J[ IJindexTeller ] = edgeDof[1] # y index
		X[ IJindexTeller ] = -T[1,1] # verdi

		IJindexTeller = IJindexTeller+1
		# celle z1 bort, x1 ned
		#33
		I[ IJindexTeller ] = edgeDof[4] # x index
		J[ IJindexTeller ] = edgeDof[2] # y index
		X[ IJindexTeller ] = -T[1,2] # verdi

		IJindexTeller = IJindexTeller+1
		# celle x1 bort, x1 ned
		#34
		I[ IJindexTeller ] = edgeDof[5] # x index
		J[ IJindexTeller ] = edgeDof[0] # y index
		X[ IJindexTeller ] = -T[2,0] # verdi

		IJindexTeller = IJindexTeller+1
		# celle y1 bort, x1 ned
		#35
		I[ IJindexTeller ] = edgeDof[5] # x index
		J[ IJindexTeller ] = edgeDof[1] # y index
		X[ IJindexTeller ] = -T[2,1] # verdi

		IJindexTeller = IJindexTeller+1
		# celle z1 bort, x1 ned
		#36
		I[ IJindexTeller ] = edgeDof[5] # x index
		J[ IJindexTeller ] = edgeDof[2] # y index
		X[ IJindexTeller ] = -T[2,2] # verdi
		
		IJindexTeller = IJindexTeller+1

	#KK = sparse(I,J,X,kDof,kDof) # TODO  Shape: (236919, 236919) 2D symmetrical
	#KK = spar.csr_matrix((I,J,X),shape=(kDof,kDof)) # csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])

	KK = csr_matrix((X, (I,J)), shape=(kDof,kDof))
	#print(KK)
	#print(KK.toarray())
	#KK = pysparse(I,J,X,kDof,kDof)
	#c = find(extC==0);
	c = np.argwhere(extC==0)
	#KK = KK(bb,bb);
	#print(KK[0].toarray())
	
	#KK[c[0,:],:] = 0
	#print(KK[0].toarray())
	#print(c)
	#KK = np.delete(KK, c[:,0])
	#print(KK.toarray())
	KK = KK.tolil()
	
	for i in range(kDof):
		for j in range(kDof):
			if np.abs(KK[i,j]) < 0.1:
				KK[i,j] = 0
	#print(KK)

	KK[c[:,0],:] = 0
	KK[:,c[:,0]] = 0
	extF[c[:,0]] = 0
	
	#print(c[:,0])	
	#print(KK)
	#print(KK.shape)
	#print(np.shape(extF))
	#print(np.shape(c[:,0]))

	#KK = KK.tocsr()
	try:
		#U = np.linalg.solve(KK, extF) # nodal displ = stiffness / force
		#U = scipy.sparse.linalg.spsolve(KK, extF)	
		U = np.linalg.lstsq(KK.toarray(), extF, rcond=None)[0] # nodal displ = stiffness / force
		# setter tilbake rader og kolonner som er fjernet grunnet boundary
		#cc = find(extC==1); # setter bare inn pï¿½ indekser der det er noe, ellers 0
		cc = np.argwhere(extC==1)
		Ufull=np.zeros(numberOfNodes*3)
		#print(np.shape(cc), np.shape(U))
		for i in range(len(cc)):
			Ufull[cc[i,0]] = U[i]
		

		#dU = np.reshape(Ufull, 3, [])
		dU = np.reshape(Ufull, (-1,3))

	except np.linalg.LinAlgError:
		Ufull=np.ones(numberOfNodes*3)*1000000
		dU = np.reshape(Ufull, (-1,3))
	
	dN = dU
	return dN

