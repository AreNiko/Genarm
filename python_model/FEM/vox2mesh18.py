import numpy as np
#import tensorflow as tf

#@tf.function
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

    (dimX,dimY,dimZ) = vG.shape
    E = np.zeros((np.count_nonzero(vG)*18,3))
    N = np.zeros((np.count_nonzero(vG), 3))
    vGvokselNr = np.zeros((dimX,dimY,dimZ))

    i = 0
    j = 0
    for z in range(1,dimZ-1):
        for y in range(1,dimY-1):
            for x in range(1,dimX-1):
                
                if vG[x,y,z] == 1: # jeg er en voxel
                    N[j,:] = (x, y, z)  # legger inn node coordinat
                    vGvokselNr[x,y,z] = j
                    j = j + 1

                    if vG[x,y,z-1] == 1: # Fins en vox rett ned
                        E[i,0] = vGvokselNr[x,y,z]
                        E[i,1] = vGvokselNr[x,y,z-1]
                        i = i + 1

                    if vG[x-1,y,z-1] == 1: # Fins en vox ned/bak i x kjørertning
                        E[i,0] = vGvokselNr[x,y,z]
                        E[i,1] = vGvokselNr[x-1,y,z-1]
                        i = i + 1

                    if vG[x+1,y,z-1] == 1: # Fins en vox ned/forran i x kjørertning
                        E[i,0] = vGvokselNr[x,y,z]
                        E[i,1] = vGvokselNr[x+1,y,z-1]
                        i = i + 1

                    if vG[x,y-1,z-1] == 1: # Fins en vox ned/høyre i x kjørertning
                        E[i,0] = vGvokselNr[x,y,z]
                        E[i,1] = vGvokselNr[x,y-1,z-1]
                        i = i + 1

                    if vG[x,y+1,z-1] == 1: # Fins en vox ned/venstre i x kjørertning
                        E[i,0] = vGvokselNr[x,y,z]
                        E[i,1] = vGvokselNr[x,y+1,z-1]
                        i = i + 1

                    if vG[x-1,y,z] == 1: # Fins en vox samme plan bak i x kjørertning
                        E[i,0] = vGvokselNr[x,y,z]
                        E[i,1] = vGvokselNr[x-1,y,z]
                        i = i + 1

                    if vG[x,y-1,z] == 1: # Fins en vox samme plan høyre i x kjørertning
                        E[i,0] = vGvokselNr[x,y,z]
                        E[i,1] = vGvokselNr[x,y-1,z]
                        i = i + 1

                    if vG[x-1,y-1,z] == 1: # Fins en vox samme plan bak/høyre i x kjørertning
                        E[i,0] = vGvokselNr[x,y,z]
                        E[i,1] = vGvokselNr[x-1,y-1,z]
                        i = i + 1

                    if vG[x+1,y-1,z] == 1: # Fins en vox samme plan forran/høyre i x kjørertning
                        E[i,0] = vGvokselNr[x,y,z]
                        E[i,1] = vGvokselNr[x+1,y-1,z]
                        i = i + 1

    E = E[0:i,:]

    #if marginsAreAdded:
    #    N = N - 1
    return E,N,vGvokselNr
"""

function [E,N,vGvokselNr] = vox2mesh18(vG)
% Lager en type 18 FEM mesh fra vokselkropp
% vGvokselNr er en 3d array som matcher vG, inneholdene
% nummeret til akruell voksel (ikke 1/0)
% Voksler telles i stigende rekkefølge ZYX posisjon i vG

for z=2:dimZ-1
    for y=2:dimY-1
        for x=2:dimX-1
            if vG(x,y,z) == 1 % jeg er en voxel
                
                
                N(j,:) = [x y z]; % legger inn node coordinat
                vGvokselNr(x,y,z) = j;
                j = j + 1;
                
                %%%%%%%%%%%%%%%% Lager edger
                
                if vG(x,y,z-1) == 1 % Fins en vox rett ned
                    E(i,1) = vGvokselNr(x,y,z);
                    E(i,2) = vGvokselNr(x,y,z-1);
                    i = i + 1;
                end
                
                if vG(x-1,y,z-1) == 1 % Fins en vox ned/bak i x kjørertning
                    E(i,1) = vGvokselNr(x,y,z);
                    E(i,2) = vGvokselNr(x-1,y,z-1);
                    i = i + 1;
                end
                
                if vG(x+1,y,z-1) == 1 % Fins en vox ned/forran i x kjørertning
                    E(i,1) = vGvokselNr(x,y,z);
                    E(i,2) = vGvokselNr(x+1,y,z-1);
                    i = i + 1;
                end
                
                if vG(x,y-1,z-1) == 1 % Fins en vox ned/høyre i x kjørertning
                    E(i,1) = vGvokselNr(x,y,z);
                    E(i,2) = vGvokselNr(x,y-1,z-1);
                    i = i + 1;
                end
                
                if vG(x,y+1,z-1) == 1 % Fins en vox ned/venstre i x kjørertning
                    E(i,1) = vGvokselNr(x,y,z);
                    E(i,2) = vGvokselNr(x,y+1,z-1);
                    i = i + 1;
                end
                
                %                 if vG(x+1,y+1,z-1) == 1 % midtkryss, ned fram i yx kjï¿½rertning
                %                     E(i,1) = voxNrMtx(x,y,z);
                %                     E(i,2) = voxNrMtx(x+1,y+1,z-1);
                %                     i = i + 1;
                %                 end
                
                %                 if vG(x-1,y-1,z-1) == 1 % midtkryss, ned bak i yx kjï¿½rertning
                %                     E(i,1) = voxNrMtx(x,y,z);
                %                     E(i,2) = voxNrMtx(x-1,y-1,z-1);
                %                     i = i + 1;
                %                 end
                
                %                 if vG(x+1,y-1,z-1) == 1 % midtkryss, ned fram i x kjï¿½rertning, ned bak i y kjï¿½rertning
                %                     E(i,1) = voxNrMtx(x,y,z);
                %                     E(i,2) = voxNrMtx(x+1,y-1,z-1);
                %                     i = i + 1;
                %                 end
                
                %                 if vG(x-1,y+1,z-1) == 1 % midtkryss, ned bak i x kjï¿½rertning, ned fram i y kjï¿½rertning
                %                     E(i,1) = voxNrMtx(x,y,z);
                %                     E(i,2) = voxNrMtx(x-1,y+1,z-1);
                %                     i = i + 1;
                %                 end
                
                if vG(x-1,y,z) == 1 % Fins en vox samme plan bak i x kjørertning
                    E(i,1) = vGvokselNr(x,y,z);
                    E(i,2) = vGvokselNr(x-1,y,z);
                    i = i + 1;
                end
                
                if vG(x,y-1,z) == 1 % Fins en vox samme plan høyre i x kjørertning
                    E(i,1) = vGvokselNr(x,y,z);
                    E(i,2) = vGvokselNr(x,y-1,z);
                    i = i + 1;
                end
                
                if vG(x-1,y-1,z) == 1 % Fins en vox samme plan bak/høyre i x kjørertning
                    E(i,1) = vGvokselNr(x,y,z);
                    E(i,2) = vGvokselNr(x-1,y-1,z);
                    i = i + 1;
                end
                
                if vG(x+1,y-1,z) == 1 % Fins en vox samme plan forran/høyre i x kjørertning
                    E(i,1) = vGvokselNr(x,y,z);
                    E(i,2) = vGvokselNr(x+1,y-1,z);
                    i = i + 1;
                end
                %
                %                 if vG(x,y,z-1) == 1 % rett ned
                %                     E(i,1) = voxNrM(x,y,z);
                %                     E(i,2) = voxNrM(x,y,z-1);
                %                     i = i + 1;
                %                 end
                
                %                 if vG(x-1,y,z-1) == 1 % veggkryss, bak i x kjï¿½rertning
                %                     E(i,1) = voxNrM(x,y,z);
                %                     E(i,2) = voxNrM(x-1,y,z-1);
                %                     i = i + 1;
                %                 end
                
            end
        end
    end
end

E = E(1:i-1,:);

if marginsAreAdded
   N = N - 1;  
end

end
"""
