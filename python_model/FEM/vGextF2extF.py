import numpy as np

def vGextF2extF(vGextF,vG):
    extF = np.zeros((int(np.sum(vG)),3))

    nodeNr = 0 # Denne må matche rekkefølgen på de genererte edgene, dvs vi itererer først z, så y, så x under
    for z in range(vG.shape[2]):
        for y in range(vG.shape[1]):
            for x in range(vG.shape[0]):
                if vG[x,y,z] == 1:
                    if vGextF[x,y,z] != 0:
                        extF[nodeNr,:] = vGextF[x,y,z] # Kunne akkululert en index og ungått vGnodeNr, men vil synliggjøre
                    nodeNr = nodeNr + 1
    return extF
