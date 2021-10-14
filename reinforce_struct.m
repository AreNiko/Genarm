function vG = reinforce_struct(vG, vGextC, vGextF, vGstayOff, noVoxToRemove)

    [E,N, ~] = vox2mesh18(vG);
    radius = 0.003; E(:,3) = pi*radius^2;

    extC = vGextC2extC(vGextC,vG);
    extF = vGextF2extF(vGextF,vG);

    extF = extF.* [0 1 0];
        
    extC(:,3) = 0;

    [sE, dN] = FEM_truss(N,E, extF,extC); % The most taxing process 

    Nstress = edgeStress2nodeStress(N,E, sE);
    NairStress = nodeStress2airNodeStress(Nstress,vG);
    Nstress = cutStayOffNodes(Nstress, vGstayOff);
    NairStress = cutStayOffNodes(NairStress, vGstayOff);
    NstressSorted = sortrows(Nstress,4,'ascend');
    NairStressSorted = sortrows(NairStress,4,'descend');
    
    vG = safeRemoveVoxels3D(vG, NstressSorted, noVoxToRemove);
    for n=1:noVoxToRemove
        vG( NairStressSorted(n,1), NairStressSorted(n,2), NairStressSorted(n,3) ) = 1;
    end

    vG = gravitate2D(vG, vGstayOff); vG = gravitate2D(vG, vGstayOff);
    