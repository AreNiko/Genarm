clear; close all;
%Attempt to grow a tree from the ground and up
rng(2);
M = 100;
N = 100;
K = 100;
nodes = 1000000;
figcount = 1;
origo_x = round(M/2);
origo_y = round(N/2);
origo_z = 2;
vG = zeros(M, N, K,'int8');
gradgrow = zeros(M, N, K);
save_locked = vG;
save_force = vG;
x = round(M/2); y = round(N/2); z = origo_z+1;
amount_t1 = 0;
amount_t0 = 0;
noVoxToRemove = 10;
step_count = 0;
coord_list = [];
gener = 1;

vG(x-1:x+1,y-1:y+1,z-1:z+1) = 1;

while amount_t1 < nodes
    fprintf("Gen %d\n", gener);
    gener = gener + 1;
    step_count = step_count + 1;
    amount_t0 = amount_t1;
    dim = size(vG);
    x_mid = 0;
    y_mid = 0;
    z_mid = 0;
    for i = 2:dim(1)-1
        for j = 2:dim(2)-1
            for k = 2:dim(3)-1
                point = vG(i,j,k);
                if point ~= 0
                    x_mid = x_mid + i;
                    y_mid = y_mid + j;
                    z_mid = z_mid + k;
                end
                surr = vG(i-1,j,k)+vG(i+1,j,k)+vG(i,j-1,k)+vG(i,j+1,k)+vG(i,j,k-1)+vG(i,j,k+1)+vG(i,j,k);
                
                if surr == 7
                    gradgrow(i,j,k) = 0;
                elseif surr == 0
                    gradgrow(i,j,k) = 0;
                else
                    gradgrow(i,j,k) = double(surr)/14;
                end
                
            end
        end
    end
    x_mid = x_mid/nnz(vG);
    y_mid = y_mid/nnz(vG);
    z_mid = z_mid/nnz(vG);
    coord_list = [coord_list; x_mid y_mid z_mid];
    disp([x_mid, y_mid, z_mid]);
    
    for i = 1:dim(1)
        for j = 1:dim(2)
            for k = 1:dim(3)
                if 0 ~= gradgrow(i,j,k)
                    cp = sqrt((i-origo_x)*(i-origo_x)+(j-origo_y)*(j-origo_y));
                    cp_avg = 1 - cp/dim(1);
                    
                    gradgrow(i,j,k) = gradgrow(i,j,k)*cp_avg*cp_avg*cp_avg*cp_avg*cp_avg;
                end
            end
        end
    end
    gradgrow(gradgrow<0) = 0;
    %fprintf('(%d, %d, %d)\n', x, y, z);
    coord_list = [coord_list; x y z];
    vG(x-1:x+1,y-1:y+1,z-1:z+1) = 1;
    %vG(:,:,1:2) = 0;
    
    % Center of mass
    for i = 1:dim(1)
        for j = 1:dim(2)
            for k = 1:dim(3)
                check = rand;
                if check < gradgrow(i,j,k)
                    
                    vG(i-1,j,k) = 1; 
                    vG(i+1,j,k) = 1;
                    vG(i,j-1,k) = 1; 
                    vG(i,j+1,k) = 1;
                    vG(i,j,k-1) = 1;
                    vG(i,j,k+1) = 1;
                    vG(i,j,k)   = 1;
                    
                    %vG(i-1:i+1,j-1:j+1,k-1:k+1) = 1;
                else
                    if check > 0.7
                        vG(i,j,k) = 0;
                    end
                end
                
            end
        end
    end
    for i=1:K
        if sum(vG(:,:,i), 'all') ~= 0
            z_top = i;
        else
            break
        end
    end
    vGextC = zeros(size(vG),'int8');
    vGextC(:,:,origo_z) = vG(:,:,origo_z); 
    
    vGextF = zeros(size(vG),'int8');
    vGextF(:,:,z_top) = vG(:,:,z_top);
    vGextF = vGextF + save_locked;
    vGextF(vGextF>1) = 1;
    
    vGstayOff = zeros(size(vG),'int8');
    vGstayOff(:,:,z_top) = vG(:,:,z_top);
    vGstayOff = vGstayOff + save_locked;
    vGstayOff(vGstayOff>1) = 1;
    
    figure(figcount);clf;plotVg_safe(vG,'edgeOff');
    hold on;plotVg_safe(vGstayOff,'edgeOff','col',[0.9 0.9 0.5]);
    hold on;plotVg_safe(vGextF,'edgeOff','col',[0.5 0.5 0.5]);
    hold on;plotVg_safe(vGextC,'edgeOff','col',[0.6 0.6 0.8]);
    title("(Stayoff (Gul), extF (grå), låst (blå)");
    sun1;
    
    saveFigToAnimGif('growing_arm.gif', step_count==1);
    
    amount_t1 = sum(vG,'all');
    fprintf('Amount of nodes: %d', amount_t1);
    
    if amount_t1 == 10
        %noVoxToRemove = noVoxToRemove + 1;
        noVoxToRemove = round(amount_t1/10);
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
        extF = vGextF2extF(vGextF,vG);
        
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

        vG = safeRemoveVoxels3D(vG, NstressSorted, noVoxToRemove);
        
        for n=1:noVoxToRemove
           vG( NairStressSorted(n,1), NairStressSorted(n,2), NairStressSorted(n,3) ) = 1;
        end

        vG = gravitate2D(vG, vGstayOff); vG = gravitate2D(vG, vGstayOff);
    end
    %{
    new_branch = rand;
    if new_branch < 0.01 && z > 10
        save_locked(x,y,z+1) = vG(x,y,z+1);
        save_force(x,y,z) = vG(x,y,z);
        
        coord_list = unique(coord_list,'rows');
        for amhere =1:length(coord_list)
            if vG(coord_list(amhere)) == 0
                first = coord_list(1:amhere-1);
                last = coord_list(amhere+1:end);
                coord_list = [first; last];
            end
        end
        
        prev_line = coord_list(randi(length(coord_list)), :);
        x = prev_line(1); y = prev_line(2); z = prev_line(3);
    end
    %}
    fprintf("\n");
end