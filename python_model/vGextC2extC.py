import numpy as np

def vGextC2extC(vGextC,vG):
    extC = np.ones((np.count_nonzero(vG),3))

    nodeNr = 0 # Denne må matche rekkefølgen på de genererte edgene, dvs vi itererer først z, så y, så x under
    for z in range(len(vGextC[:,3])):
        for y in range(len(vGextC[:,2])):
            for x in range(len(vGextC[:,1])):
                if vG[x,y,z] == 1:
                    if vGextC[x,y,z] == 1:
                        extC[nodeNr,:] = [0,0,0] 
                    nodeNr = nodeNr + 1
    return extC
