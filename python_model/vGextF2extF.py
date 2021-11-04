import numpy as np

def vGextF2extF(vGextF,vG):
    extF = np.ones((np.count_nonzero(vG),3))

    nodeNr = 0 # Denne må matche rekkefølgen på de genererte edgene, dvs vi itererer først z, så y, så x under
    for z in range(len(vGextF[:,3])):
        for y in range(len(vGextF[:,2])):
            for x in range(len(vGextF[:,1])):
                if vG[x,y,z] == 1:
                    if vGextF[x,y,z] != 0:
                        extF[nodeNr,:] = vGextF[x,y,z] # Kunne akkululert en index og ungått vGnodeNr, men vil synliggjøre
                    nodeNr = nodeNr + 1
    return extF