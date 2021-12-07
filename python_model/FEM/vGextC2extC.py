import numpy as np

def vGextC2extC(vGextC,vG):
    extC = np.ones((int(np.sum(vG)),3))

    nodeNr = 0 # Denne må matche rekkefølgen på de genererte edgene, dvs vi itererer først z, så y, så x under
    for z in range(vG.shape[2]):
        for y in range(vG.shape[1]):
            for x in range(vG.shape[0]):
                #print(x,y,z)
                if vG[x,y,z] == 1:
                    if vGextC[x,y,z] == 1:
                        extC[nodeNr,:] = (0,0,0) 
                    nodeNr += 1
    return extC
