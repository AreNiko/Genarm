function [vG, sE, dN] = reinforce_struct(vG, vGextC, vGextF, vGstayOff, noVoxToRemove)

    [E,N, ~] = vox2mesh18(vG);
    radius = 0.003; E(:,3) = pi*radius^2;

    extC = vGextC2extC(vGextC,vG);
    extF = vGextF2extF(vGextF,vG);
    randirect = randi([0,360]);
    extF = extF.* [100*cosd(randirect) 100*sind(randirect) 0];
    
    extC(:,3) = 0;

    [sE, dN] = FEM_truss(N, E, extF, extC); 
    %disp(dN)
    %disp(max(abs(dN),[],'all'))
    %pause
    Nstress = edgeStress2nodeStress(N, E, sE);
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
    
    figure(1);clf;plotVg_safe(vG,'edgeOff');
    hold on;plotVg_safe(vGstayOff,'edgeOff','col',[0.9 0.9 0.5]);
    hold on;plotVg_safe(vGextF,'edgeOff','col',[0.5 0.5 0.5]);
    hold on;plotVg_safe(vGextC,'edgeOff','col',[0.6 0.6 0.8]);
    title("Stay off (gul), extF (grå), låst (blå)");
    sun1;
   