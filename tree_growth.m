clear; close all;
%Attempt to grow a tree from the ground and up
%rng(2);
M = 100;
N = 100;
K = 100;
nodes = 500000;
figcount = 1;
origo_z = 2;
vG = zeros(M, N, K,'int8');
save_locked = vG;
save_force = vG;
x = round(M/2); y = round(N/2); z = origo_z+1;
amount_t1 = 0;
amount_t0 = 0;
noVoxToRemove = 10;
step_count = 0;
coord_list = [];
gener = 1;
while amount_t1 < nodes
    fprintf("Gen %d\n", gener);
    gener = gener + 1;
    step_count = step_count + 1;
    amount_t0 = amount_t1;
    
    direct = rand;
    if direct < 0.22 % Left
        x = x - 1;
    elseif 0.22 < direct && direct < 0.44 % Forward
        y = y + 1;
    elseif 0.44 < direct && direct < 0.66 % Right
        x = x + 1;
    elseif 0.66 < direct && direct < 0.88  % Back
        y = y - 1;
    else % Up
        z = z + 1;
    end
    %fprintf('(%d, %d, %d)\n', x, y, z);
    coord_list = [coord_list; x y z];
    vG(x-1:x+1,y-1:y+1,z-1:z+1) = 1;
    %vG(:,:,1:2) = 0;
    vGextC = zeros(size(vG),'int8');
    vGextC(:,:,origo_z) = vG(:,:,origo_z); 
    
    vGextF = zeros(size(vG),'int8');
    vGextF(x-2:x+2,y-2:y+2,z) = vG(x-2:x+2,y-2:y+2,z);
    vGextF = vGextF + save_locked;
    vGextF(vGextF>1) = 1;
    
    vGstayOff = zeros(size(vG),'int8');
    vGstayOff(x-1:x+1,y-1:y+1,z+1) = vG(x-1:x+1,y-1:y+1,z+1);
    vGstayOff = vGstayOff + save_locked;
    vGstayOff(vGstayOff>1) = 1;
    
    figure(figcount);clf;plotVg_safe(vG,'edgeOff');
    hold on;plotVg_safe(vGstayOff,'edgeOff','col',[0.9 0.9 0.5]);
    hold on;plotVg_safe(vGextF,'edgeOff','col',[0.5 0.5 0.5]);
    hold on;plotVg_safe(vGextC,'edgeOff','col',[0.6 0.6 0.8]);
    title("(Stayoff (Gul), extF (grå), låst (blå)");
    sun1;
    
    saveFigToAnimGif('tree_growing.gif', step_count==1);
    
    amount_t1 = sum(vG,'all');
    fprintf('Amount of nodes: %d', amount_t1);
    
    if amount_t0 == amount_t1
        %noVoxToRemove = noVoxToRemove + 1;
        fprintf(', Amount of nodes to displace: %d', noVoxToRemove);
        %figure(figcount+1);clf;plotVg_safe(vG,'edgeOff');
        %hold on;plotVg_safe(vGstayOff,'edgeOff','col',[0.9 0.9 0.5]);
        %hold on;plotVg_safe(vGextF,'edgeOff','col',[0.5 0.5 0.5]);
        %hold on;plotVg_safe(vGextC,'edgeOff','col',[0.6 0.6 0.8]);
        %title("extF (grå), låst (blå)");
        %sun1;
        
        [E,N, ~] = vox2mesh18(vG);
        radius = 0.003; E(:,3) = pi*radius^2;
        extC = vGextC2extC(vGextC,vG);
        extF = vGextF2extF(vGextF,vG) .* [0 0 10];
        %{
        rand_n = rand;
        if rand_n < 0.25
            extF = vGextF2extF(vGextF,vG) .* [0 10 0];
        elseif 0.25 < rand_n < 0.5
            extF = vGextF2extF(vGextF,vG) .* [10 0 0];
        elseif 0.5 < rand_n < 0.75
            extF = vGextF2extF(vGextF,vG) .* [0 -10 0];
        else 
            extF = vGextF2extF(vGextF,vG) .* [-10 0 0];
        end
        %}
        %extC(:,3) = 0;
        %{
        findsing = 0;
        while findsing < 1
            try 
                issing = inv(extF);
                findsing = 1;
            catch
                crea = eye(size(extF));
                extF = extF + crea.*eye(6)*1e-3;
            end
        end
        %}
        [sE, dN] = FEM_truss2(N,E, extF,extC);
        
        Nstress  = edgeStress2nodeStress(N,E, sE);
        NairStress  = nodeStress2airNodeStress(Nstress,vG);
        Nstress = cutStayOffNodes(Nstress, vGstayOff);
        NairStress = cutStayOffNodes(NairStress, vGstayOff);
        NstressSorted = sortrows(Nstress,4,'ascend');
        NairStressSorted = sortrows(NairStress,4,'descend');

        %vG = safeRemoveVoxels3D(vG, NstressSorted, noVoxToRemove);
        
        for n=1:noVoxToRemove
           vG( NairStressSorted(n,1), NairStressSorted(n,2), NairStressSorted(n,3) ) = 1;
        end

        vG = gravitate2D(vG, vGstayOff); vG = gravitate2D(vG, vGstayOff);
    end
    new_branch = rand;
    if new_branch < 0.01 && z > 10
        save_locked(x,y,z+1) = vG(x,y,z+1);
        save_force(x,y,z) = vG(x,y,z);
        
        coord_list = unique(coord_list,'rows');
        %{
        for amhere =1:length(coord_list)
            if vG(coord_list(amhere)) == 0
                first = coord_list(1:amhere-1);
                last = coord_list(amhere+1:end);
                coord_list = [first; last];
            end
        end
        %}
        
        prev_line = coord_list(randi(length(coord_list)), :);
        x = prev_line(1); y = prev_line(2); z = prev_line(3);
    end
    fprintf("\n");
end




%{
% TODO: Generalize this to work for different parameters
vG_work = vG;
nnz(vG_work)
f_layer = origo(3) + arm_height + height - thickness;
b_layer = origo(3) + arm_height + max_radius - thickness*2;

%vG_work(:,:,1:b_layer) = 0;
%vG_work(:,:,f_layer:end) = 0;

%%%%%%%%%%% Definerer låste voksler - blå
vGextC = zeros(size(vG_work),'int8');
%vGc_1more = outerShell(vGc, ceil(thickness/2));
vGextC(:,:,1:b_layer+1) = vG_work(:,:,1:b_layer+1); 
%vGextC = vGc_shell.*vG_work;

%%%%%%%%%%% Definerer kraft voksler - grå
vGextF = zeros(size(vG_work),'int8');
vGextF(:,:,f_layer:end) = vG_work(:,:,f_layer:end);

%%%%%%%%%%% Definerer stayOff voksler - gule
%vGstayOff = zeros(size(vG_work),'int8');
%vGstayOff(:,:,f_layer:end) = vG_work(:,:,f_layer:end); 
vGstayOff = zeros(size(vG_work),'int8');
vGstayOff = vGcylinder(vGstayOff,[origo(1),origo(2),f_layer],1, wrist_radius,6);

vGextF = vGextF - vGstayOff;
vGextF(vGextF<0) = 0;

noVoxToRemove = 200; % for eksempel
vGsave = vG_work;
% Generative steps
gt = 100;
for i = 1:gt
    tic
    fprintf("Step: %d/%d \n", i, gt)
    figure(figcount);clf;plotVg_safe(vG_work,'edgeOff');
    hold on;plotVg_safe(vGstayOff,'edgeOff','col',[0.9 0.9 0.5]);
    hold on;plotVg_safe(vGextF,'edgeOff','col',[0.5 0.5 0.5]);
    hold on;plotVg_safe(vGextC,'edgeOff','col',[0.6 0.6 0.8]);
    title("Stay off (gul), extF (grå), låst (blå)");
    sun1;
    
    saveFigToAnimGif('roboarmfinal.gif', i==1);
    
    [E,N, ~] = vox2mesh18(vG_work);
    radius = 0.003; E(:,3) = pi*radius^2;
    midpoint = [origo(1) origo(2) f_layer-1];
    extC = vGextC2extC(vGextC,vG_work);
    extF = vGextF2extF(vGextF,vG_work);
    idx = mod(i,4);
    if idx == 0
        extF = extF.* [100 0 0];
    elseif idx == 1
        extF = extF.* [0 100 0];
    elseif idx == 2
        extF = extF.* [-100 0 0];
    else
        extF = extF.* [0 -100 0];
    end
    
    extC(:,3) = 0;
    %{
    figure(figcount+1);clf;plotMesh(N,E,'lTykk',0.01);
    hold on;plotMeshCext(N,extC,1);
    hold on;plotMeshFext(N,extF, 0.01);
    %}
    [sE, dN] = FEM_truss(N,E, extF,extC); % The most taxing process 
    
    %visuellScale = 100;
    %figure(figcount+1);clf;plotMesh(N-visuellScale*dN,E,'lTykk',0.01); % Mesh med nodeforflytninger
    
    Nstress  = edgeStress2nodeStress(N,E, sE);
    NairStress  = nodeStress2airNodeStress(Nstress,vG_work);
    Nstress = cutStayOffNodes(Nstress, vGstayOff);
    NairStress = cutStayOffNodes(NairStress, vGstayOff);
    NstressSorted = sortrows(Nstress,4,'ascend');
    NairStressSorted = sortrows(NairStress,4,'descend');
    
    vG_work = safeRemoveVoxels3D(vG_work, NstressSorted, noVoxToRemove);
    
    for n=1:noVoxToRemove
       vG_work( NairStressSorted(n,1), NairStressSorted(n,2), NairStressSorted(n,3) ) = 1;
    end
    
    %vG_work = gravitate2D(vG_work, vGstayOff); vG_work = gravitate2D(vG_work, vGstayOff);
    toc
end
figcount = figcount+1;
figcount = figcount+1;figure(figcount);clf;plotVg_safe(vG_work,'edgeOff');
hold on;plotVg_safe(vGstayOff,'edgeOff','col',[0.9 0.9 0.5]);
hold on;plotVg_safe(vGextF,'edgeOff','col',[0.5 0.5 0.5]);
hold on;plotVg_safe(vGextC,'edgeOff','col',[0.6 0.6 0.8]);
title("Stay off (gul), extF (grå), låst (blå)");
sun1;
%}