function [vG, figcount] = gengrow(env, origo, arm_radiusx, arm_radiusy, wrist_radius, height, thickness, noVoxToRemove, weight_hand, weight_want, arm_height, figcount)
% inputs
%variables:
    %arm_radiusx:   Radius of wanted arm in x direction
    %arm_radiusy:   Radius of wanted arm in y direction
    %wrist_radius:  Wanted radius of wrist
    %height:        Wanted length/height of prosthesis
    %thickness:     Thickness of shell around scanned arm
    %gt:            Amount of generations of generative process
    %noVoxToRemove: Number of voxels to remove for each generation
    %weight_hand: Weight of hand prosthetic;
    %weight_want: Wanted weight of entire prosthetic;
    %figcount: Which number figure it should start from
    
    %Temporary test arm variables:
        %arm_height: Length/height of test arm
    
    PLAdens = 1.24; %Density of PLA in g/cm^3
    nozzwidth = 0.2; %Nozzle width of 3D printer in cm
    infill = 0.2; % Infill of 3d printer, for now set to 20%
    Voxsize = 0.2; % FOR NOW SET VOX SIZE TO 1mm / 0.1 cm
    plainfill = (Voxsize*Voxsize*Voxsize*infill*0.001) + (Voxsize*Voxsize*6*0.01); % Area of a voxel in cm^3 with infill% infill % Not taking into account that the voxels are connected yet
    

    max_radius = max(arm_radiusx, arm_radiusy);
    width = (arm_radiusx+thickness)*2;
    depth = (arm_radiusy+thickness)*2;

    vG = vGcone(env, [origo(1) origo(2) origo(3)+arm_height], width, depth, height, wrist_radius);
    figcount=figcount+1;figure(figcount);clf;plotVg_safe(vG, 'edgeOff');
    stayoff_shell1 = outerShell(vG, 1);
    %figure(2);clf;plotMesh(N,E);
    % INSERT CALCULATES FOR FINDING THE LARGEST POSSIBLE radius    % Use the largest possible width distance as the radius of the arm 
    % Placeholder for arm connection point

    vGc = env;
    for i = 1:arm_height
        vGc = vGsphere(vGc, [origo(1) origo(2) origo(3)+i], 1, max_radius);
        vGc = vGsphere(vGc, [origo(1)+i origo(2) origo(3)], 1, max_radius);
    end

    % vGc = innerFillet(vGc,10);
    % figcount=figcount+1;figure(figcount);clf;plotVg_safe(vGc, 'edgeOff');

    vGc_shell = outerShell(vGc, thickness);
    vG = vG + vGc_shell;
    vG(vG>1) = 1;

    vG = innerShell(vG,thickness) - vGc;
    vGc1 = shift_3dmatrix(vGc, round((thickness+max_radius)/4),0,0);

    vG = vG - vGc1;
    vG(vG<0) = 0;

    vG(origo(1) + (max_radius+thickness):end,:,:) = 0;
    figcount=figcount+1;figure(figcount);clf;plotVg_safe(vG, 'edgeOff');
    %[N, E, F, C, vGno] = make_N_E_F_C_vGno(vG, 2, 1);
    %[sE, dN] = FEM_truss(N,E, F,C);
    %figcount=figcount+1;figure(figcount);clf;plotMesh(N,E); % evt. plotMesh(N,E,'txt');

    figcount=figcount+1;
    
    current_weight = PLAdens*(plainfill*nnz(vG)*(nozzwidth/2)^2)*pi + weight_hand; 
    wanted_nrnodes = (weight_want - weight_hand)/((PLAdens*plainfill*(nozzwidth/2)^2)*pi); 
    % Add weight of biological forarm to get similar weight as an average arm?
    fprintf("Current weight of generated prosthesis versus wanted weight \n %.5f/%.5f \n", current_weight, weight_want);
    %%
    % TODO: Generalize this to work for different parameters
    vG_work = vG;
    f_layer = origo(3) + arm_height + height - thickness;
    b_layer = origo(3) + arm_height + max_radius - thickness;

    %vG_work(:,:,1:b_layer) = 0;
    %vG_work(:,:,f_layer:end) = 0;

    %%%%%%%%%%% Definerer låste voksler - blå
    vG_work(:,:,b_layer+1:end) = 0;
    vG_work(:,:,1:b_layer-7) = 0;
    
    vGextC = zeros(size(vG_work),'int8');
    vGextC(:,:,1:b_layer+1) = vG_work(:,:,1:b_layer+1);
    
    
    %vGc_1more = outerShell(vGc, ceil(thickness/2));
    %vGextC = vGc_shell.*vG_work;
    
    %vGextC = vGc;

    %%%%%%%%%%% Definerer kraft voksler - grå
    vGextF = zeros(size(vG_work),'int8');
    %vGextF(:,:,f_layer:end) = vG_work(:,:,f_layer:end);
    vGextF = vGcylinder(vGextF,[origo(1),origo(2),origo(3) + height + arm_height-thickness+1],1, wrist_radius,thickness);
    %%%%%%%%%%% Definerer stayOff voksler - gule
    %vGstayOff(:,:,f_layer:end) = vG_work(:,:,f_layer:end); 
    vGstayOff = zeros(size(vG_work),'int8');

    %vGstayOff = vGcylinder(vGstayOff,[origo(1),origo(2),origo(3) + round(thickness/2) + arm_height],1, wrist_radius,height);
    vGstayOff = vGcylinder(vGstayOff,[origo(1),origo(2),origo(3) + height + arm_height-thickness+1],1, wrist_radius,thickness);
    %vGstayOff = vGstayOff + stayoff_shell1;

    %vGextF = vGextF - vGstayOff;
    %vGextF(vGextF<0) = 0;
    
    start_coord = [];
    x = origo(1)-(max_radius+1)*cosd(0); y = origo(2)-(max_radius+1)*sind(0); z = b_layer+2;
    origo_z = b_layer+1;
    
    amount_t1 = sum(vG_work,'all');
    x_bendp = [];
    x_bendm = [];
    y_bendp = [];
    y_bendm = [];
    figcount=figcount+1;figure(figcount);clf;plotVg_safe(vG_work,'edgeOff');
    sun1;
    
    save_locked = env;
    save_force = env;
    
    step_count = 0;
    coord_list = [];
    gener = 1;
    while amount_t1 < wanted_nrnodes
        fprintf("Gen %d\n", gener);
        gener = gener + 1;
        step_count = step_count + 1;
        amount_t0 = amount_t1;
        check = 0;
        while check == 0
            direct = rand;
            if direct < 0.20 % Left
                if x - 1 > origo(1)-(max_radius+5)
                    x = x - 1;
                    check = 1;
                end
            elseif 0.20 < direct && direct < 0.40 % Forward
                if y + 1 < origo(2)+(max_radius+5)
                    y = y + 1;
                    check = 1;
                end
            elseif 0.4 < direct && direct < 0.7 % Right
                if x + 1 < origo(1)+(max_radius+5)
                    x = x + 1;
                    check = 1;
                end
            elseif 0.7 < direct && direct < 0.9  % Back
                if y-1 > origo(2)-(max_radius+5)
                    y = y - 1;
                    check = 1;
                end
            else % Up
                if z+1 < f_layer
                    z = z + 1;
                    check = 1;
                end
            end
        end
        fprintf('(%d, %d, %d)\n', x, y, z);
        coord_list = [coord_list; x y z];
        vG_work(x-1:x+1,y-1:y+1,z-1:z+1) = 1; 

        vGextF = zeros(size(vG_work),'int8');
        vGextF(x-2:x+2,y-2:y+2,z) = vG_work(x-2:x+2,y-2:y+2,z);
        %vGextF = vGextF + save_force;
        vGextF(vGextF>1) = 1;

        vGstayOff = zeros(size(vG_work),'int8');
        vGstayOff(x-1:x+1,y-1:y+1,z+1) = vG_work(x-1:x+1,y-1:y+1,z+1);
        vGstayOff = vGstayOff + vGextC;
        %vGstayOff = vGstayOff + save_locked + vGextC;
        vGstayOff(vGstayOff>1) = 1;

        figure(figcount);clf;plotVg_safe(vG_work,'edgeOff');
        hold on;plotVg_safe(vGstayOff,'edgeOff','col',[0.9 0.9 0.5]);
        hold on;plotVg_safe(vGextF,'edgeOff','col',[0.5 0.5 0.5]);
        hold on;plotVg_safe(vGextC,'edgeOff','col',[0.6 0.6 0.8]);
        title("(Stayoff (Gul), extF (grå), låst (blå)");
        sun1;

        saveFigToAnimGif('tree_growing.gif', step_count==1);

        amount_t1 = sum(vG_work,'all');
        fprintf('Amount of nodes: %d', amount_t1);

        if amount_t0 == amount_t1
            %noVoxToRemove = noVoxToRemove + 1;
            fprintf(', Amount of nodes to displace: %d', noVoxToRemove);

            [E,N, ~] = vox2mesh18(vG_work);
            radius = 0.003; E(:,3) = pi*radius^2;
            extC = vGextC2extC(vGextC,vG_work);
            extF = vGextF2extF(vGextF,vG_work);
            idx = mod(i,4);
            if idx == 0
                extF = extF.* [10 0 0];
            elseif idx == 1
                extF = extF.* [0 10 0];
            elseif idx == 2
                extF = extF.* [-10 0 0];
            else
                extF = extF.* [0 -10 0];
            end
            
            [sE, dN] = FEM_truss2(N,E, extF,extC);

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

            vG_work = gravitate2D(vG_work, vGstayOff); vG_work = gravitate2D(vG_work, vGstayOff);
        end
        
        new_branch = rand;
        if new_branch < 0.01 && z > 10
            
            save_locked(x,y,z+1) = vG_work(x,y,z+1);
            save_force(x,y,z) = vG_work(x,y,z);
            previdx = randi(length(coord_list));
            prev_line = coord_list(previdx, :);
            coord_list = coord_list(1:previdx, :);
            x = prev_line(1); y = prev_line(2); z = prev_line(3);
        end
        
        fprintf("\n");
    end
end