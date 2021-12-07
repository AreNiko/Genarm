import numpy as np
import tensorflow as tf

@tf.function
def vox2mesh18(vG):
    """
    if np.sum(vG) == 0:
        print(" - vG er tom, sorry ")
        return
    marginsAreAdded = 0
    if testForMargins(vG,1) == 0:
        print("Legger på 0-marginer \n")
        vG = addMargin(vG,1)
        marginsAreAdded = 1
    """

    (bac,dimX,dimY,dimZ) = vG.shape
    E = tf.Variable(tf.zeros([tf.reduce_sum(vG)*18,3]))
    N = tf.Variable(tf.zeros([tf.reduce_sum(vG),3]))
    #E = np.zeros((np.count_nonzero(vG)*18,3))
    #N = np.zeros((np.count_nonzero(vG), 3))
    vGvokselNr = tf.Variable(tf.zeros([dimX,dimY,dimZ]))

    i = tf.Variable(0, trainable=False)
    j = tf.Variable(0, trainable=False, dtype=tf.int32)
    j2 = tf.Variable(0.0, trainable=False)

    for z in range(1,dimZ-1):
        for y in range(1,dimY-1):
            for x in range(1,dimX-1):
                
                if vG[0,x,y,z] == 1: # jeg er en voxel
                    N[j,0].assign(x)  # legger inn node coordinat
                    N[j,1].assign(y)  # legger inn node coordinat
                    N[j,2].assign(z)  # legger inn node coordinat
                    vGvokselNr[x,y,z].assign(j2)
                    j.assign_add(1)
                    j2.assign_add(1)

                    if vG[0,x,y,z-1] == 1: # Fins en vox rett ned
                        E[i,0].assign(vGvokselNr[x,y,z])
                        E[i,1].assign(vGvokselNr[x,y,z-1])
                        i.assign_add(1)

                    if vG[0,x-1,y,z-1] == 1: # Fins en vox ned/bak i x kjørertning
                        E[i,0].assign(vGvokselNr[x,y,z])
                        E[i,1].assign(vGvokselNr[x-1,y,z-1])
                        i.assign_add(1)

                    if vG[0,x+1,y,z-1] == 1: # Fins en vox ned/forran i x kjørertning
                        E[i,0].assign(vGvokselNr[x,y,z])
                        E[i,1].assign(vGvokselNr[x+1,y,z-1])
                        i.assign_add(1)

                    if vG[0,x,y-1,z-1] == 1: # Fins en vox ned/høyre i x kjørertning
                        E[i,0].assign(vGvokselNr[x,y,z])
                        E[i,1].assign(vGvokselNr[x,y-1,z-1])
                        i.assign_add(1)

                    if vG[0,x,y+1,z-1] == 1: # Fins en vox ned/venstre i x kjørertning
                        E[i,0].assign(vGvokselNr[x,y,z])
                        E[i,1].assign(vGvokselNr[x,y+1,z-1])
                        i.assign_add(1)

                    if vG[0,x-1,y,z] == 1: # Fins en vox samme plan bak i x kjørertning
                        E[i,0].assign(vGvokselNr[x,y,z])
                        E[i,1].assign(vGvokselNr[x-1,y,z])
                        i.assign_add(1)

                    if vG[0,x,y-1,z] == 1: # Fins en vox samme plan høyre i x kjørertning
                        E[i,0].assign(vGvokselNr[x,y,z])
                        E[i,1].assign(vGvokselNr[x,y-1,z])
                        i.assign_add(1)

                    if vG[0,x-1,y-1,z] == 1: # Fins en vox samme plan bak/høyre i x kjørertning
                        E[i,0].assign(vGvokselNr[x,y,z])
                        E[i,1].assign(vGvokselNr[x-1,y-1,z])
                        i.assign_add(1)

                    if vG[0,x+1,y-1,z] == 1: # Fins en vox samme plan forran/høyre i x kjørertning
                        E[i,0].assign(vGvokselNr[x,y,z])
                        E[i,1].assign(vGvokselNr[x+1,y-1,z])
                        i.assign_add(1)

    #E = E[0:i,:]
    #if marginsAreAdded:
    #    N = N - 1
    return E,N
