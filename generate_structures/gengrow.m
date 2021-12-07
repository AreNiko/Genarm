function [vG, figcount] = gengrow(env, origo, arm_radiusx, arm_radiusy, wrist_radius, height, thickness, gt, noVoxToRemove, weight_hand, weight_want, arm_height, figcount, symmetry)
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
    vG_work(:,:,1:b_layer-3) = 0;
    
    vGextC = zeros(size(vG_work),'int8');
    vGextC(:,:,1:b_layer+1) = vG_work(:,:,1:b_layer+1);
    
    
    %vGc_1more = outerShell(vGc, ceil(thickness/2));
    %vGextC = vGc_shell.*vG_work;
    
    %vGextC = vGc;

    %%%%%%%%%%% Definerer kraft voksler - grå
    vGextF = zeros(size(vG_work),'int8');
    %vGextF(:,:,f_layer:end) = vG_work(:,:,f_layer:end);
    vGextF = vGcylinder(vGextF,[origo(1),origo(2),f_layer+1],1, wrist_radius,thickness);
    %%%%%%%%%%% Definerer stayOff voksler - gule
    %vGstayOff(:,:,f_layer:end) = vG_work(:,:,f_layer:end); 
    vGstayOff = zeros(size(vG_work),'int8');

    %vGstayOff = vGcylinder(vGstayOff,[origo(1),origo(2),origo(3) + round(thickness/2) + arm_height],1, wrist_radius,height);
    vGstayOff = vGcylinder(vGstayOff,[origo(1),origo(2),f_layer+1],1, wrist_radius,thickness);
    %vGstayOff = vGstayOff + stayoff_shell1;

    %vGextF = vGextF - vGstayOff;
    %vGextF(vGextF<0) = 0;
    
    start_coord = [];
    directprob = [];
    x = [];
    y = [];
    z = [];
    radi = floor(thickness/2)-1;
    degreesofstart = 20;
    space = 360/degreesofstart;
    
    for i = 0:space-1
        theta = i*degreesofstart;
        
        x0 = round(origo(1)-(max_radius+1)*cosd(theta));
        y0 = round(origo(2)-(max_radius+1)*sind(theta));
        %fprintf('%.5f %.5f \n', cosd(theta), sind(theta));
        %fprintf('%d %d %d \n', theta, x0, y0); 
        x = [x x0];
        y = [y y0];
        z = [z b_layer+radi];
        % Negative x direction 0.3 0.5 0.7
        % Positive y direction 0.2 0.5 0.7
        % Positive x direction 0.2 0.4 0.7
        % Negative y direction 0.2 0.4 0.6
        directprob = [directprob; 0.22 0.44 0.66 0.88];
    end
    %x = origo(1)-(max_radius+1)*cosd(0); y = origo(2)-(max_radius+1)*sind(0); z = b_layer+2;
    origo_z = b_layer+1;
    
    amount_t1 = sum(vG_work,'all');
    
    figcount=figcount+1;figure(figcount);clf;plotVg_safe(vG_work,'edgeOff');
    sun1;
    
    save_locked = env;
    save_force = env;
    
    diff = max_radius - wrist_radius;
    height_diffo = height/diff;
    
    step_count = 0;
    coord_list = [];
    gener = 1;
    %gt = 1000;
    %while amount_t1 < wanted_nrnodes
    if symmetry == "same"
        leng = 1;
    else
        leng = length(x);
    end
    
    for k = 1:1200
        fprintf("Gen %d\n", gener);
        gener = gener + 1;
        step_count = step_count + 1;
        amount_t0 = amount_t1;
        
        for i = 1:leng
            
            level = floor((z(i)-origo(3))/height_diffo);
            currentradius = max_radius - level;
            angdist = sqrt((((currentradius+thickness)*cosd(i*space))*((currentradius+thickness)*cosd(i*space))) ...
                    + (((currentradius+thickness)*sind(i*space))*((currentradius+thickness)*sind(i*space))));
            check = 0;
            while check == 0
                
                direct = rand;
                %fprintf('%d ' , direct);
                
                %if direct < 0.22 % Left
                if direct < directprob(i,1) % Left
                    if sqrt((x(i)-1-origo(1))*(x(i)-1-origo(1))+(y(i)-origo(2))*(y(i)-origo(2))) < angdist
                    %if x - 1 > origo(1)-(max_radius+5)
                        x(i) = x(i) - 1;
                        check = 1;
                        %{
                        directprob(i,1) = directprob(i,1)+0.005;
                    else
                        directprob(i,1) = directprob(i,1)-0.01;
                        %}
                    end
                    
                %elseif 0.22 < direct && direct < 0.44 % Forward
                elseif directprob(i,1) < direct && direct < directprob(i,2) % Forward
                    if sqrt((x(i)-origo(1))*(x(i)-origo(1))+((y(i)+1-origo(2))*(y(i)+1-origo(2)))) < angdist
                    %if y + 1 < origo(2)+(max_radius+5)
                        y(i) = y(i) + 1;
                        check = 1;
                        %{
                        if directprob(i,2)-directprob(i,1) < directprob(i,3)-directprob(i,2)
                            directprob(i,2) = directprob(i,2)+0.005;
                        else
                            directprob(i,1) = directprob(i,1)-0.005;
                        
                        end
                    else
                        directprob(i,1) = directprob(i,1)+0.005;
                        directprob(i,2) = directprob(i,2)-0.005;
                        %}
                    end
                    
                %elseif 0.44 < direct && direct < 0.66 % Right
                elseif directprob(i,2) < direct && direct < directprob(i,3) % Right
                    if sqrt(((x(i)+1-origo(1))*(x(i)+1-origo(1)))+((y(i)-origo(2))*(y(i)-origo(2)))) < angdist
                    %if x + 1 < origo(1)+(max_radius+5)
                        x(i) = x(i) + 1;
                        check = 1;
                        %{
                        if directprob(i,3)-directprob(i,2) < directprob(i,4)-directprob(i,3)
                            directprob(i,3) = directprob(i,3)+0.005;
                        else
                            directprob(i,2) = directprob(i,2)-0.005;
                        end
                    else
                        directprob(i,2) = directprob(i,2)+0.005;
                        directprob(i,3) = directprob(i,3)-0.005;
                        %}
                    end
                %elseif 0.66 < direct && direct < 0.88  % Back
                elseif directprob(i,3) < direct && direct < directprob(i,4) % Back
                    if sqrt(((x(i)-origo(1))*(x(i)-origo(1)))+((y(i)-1-origo(2))*(y(i)-1-origo(2)))) < angdist
                    %if y - 1 > origo(2)-(max_radius+5)
                        y(i) = y(i) - 1;
                        check = 1;
                        %{
                        directprob(i,3) = directprob(i,3)-0.005;
                    else
                        directprob(i,3) = directprob(i,3)+0.01;
                        %}
                    end
                else % Up
                    if z(i)+1 < f_layer+thickness-1
                        z(i) = z(i) + 1;
                        check = 1;
                    end
                end
            end
            %fprintf('(%d, %d, %d)\n', x, y, z);
            %coord_list = [coord_list; x y z];
            
            vG_work(x(i)-radi:x(i)+radi,y(i)-radi:y(i)+radi,z(i)-radi:z(i)+radi) = 1; 
            
            % Gray
            vGextF = zeros(size(vG_work),'int8');
            vGextF(x(i)-radi:x(i)+radi,y(i)-radi:y(i)+radi,z(i)+radi-1) = vG_work(x(i)-radi:x(i)+radi,y(i)-radi:y(i)+radi,z(i)+radi-1);
            %vGextF = vGextF + save_force;
            vGextF(vGextF>1) = 1;
            
            % Yellow
            vGstayOff = zeros(size(vG_work),'int8');
            vGstayOff(x(i)-radi:x(i)+radi,y(i)-radi:y(i)+radi,z(i)+radi) = vG_work(x(i)-radi:x(i)+radi,y(i)-radi:y(i)+radi,z(i)+radi);
            
            %vGstayOff = vGstayOff + vGextC;
            vGstayOff = vGstayOff + save_locked + vGextC;
            vGstayOff(vGstayOff>1) = 1;
            
            %{
            new_branch = rand;
            if new_branch < 0.01 && z(i) > origo_z

                save_locked(x(i),y(i),z(i)+radi) = vG_work(x(i),y(i),z(i)+radi);
                save_force(x(i),y(i),z(i)+radi-1) = vG_work(x(i),y(i),z(i)+radi-1);
                previdx = randi(length(coord_list));
                prev_line = coord_list(previdx, :);
                %coord_list = coord_list(1:previdx, :);
                x = prev_line(1); y = prev_line(2); z = prev_line(3);
            end
            %}
        end
        
        if symmetry == "same"
            for j = 0:length(x)-1
                theta = j*degreesofstart;
                hypo = sqrt((origo(1) - x(1))*(origo(1) - x(1)) + (origo(2) - y(1))*(origo(2) - y(1)));
                xd = round(origo(1) - hypo*cosd(theta));
                yd = round(origo(2) - hypo*sind(theta));
                vG_work(xd-radi:xd+radi,yd-radi:yd+radi,z(1)-radi:z(1)+radi) = 1;
            end
        end
        
        figure(figcount);clf;plotVg_safe(vG_work,'edgeOff');
        %hold on;plotVg_safe(vGstayOff,'edgeOff','col',[0.9 0.9 0.5]);
        %hold on;plotVg_safe(vGextF,'edgeOff','col',[0.5 0.5 0.5]);
        %hold on;plotVg_safe(vGextC,'edgeOff','col',[0.6 0.6 0.8]);
        title("(Stayoff (Gul), extF (grå), låst (blå)");
        sun1;

        saveFigToAnimGif('tree_growing.gif', step_count==1);

        amount_t1 = sum(vG_work,'all');
        fprintf('Amount of nodes: %d', amount_t1);
        %{
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
        %}
        fprintf("\n");
        
    end
    
    %%%%%%%%%%% Definerer låste voksler - blå
    vGextC = zeros(size(vG_work),'int8');
    vGextC(:,:,1:b_layer+1) = vG_work(:,:,1:b_layer+1);
    
    %vGc_1more = outerShell(vGc, ceil(thickness/2));
    %vGextC = vGc_shell.*vG_work;
    
    %vGextC = vGc;

    %%%%%%%%%%% Definerer kraft voksler - grå
    vGextF = zeros(size(vG_work),'int8');
    %vGextF(:,:,f_layer:end) = vG_work(:,:,f_layer:end);
    vGextF = vGcylinder(vGextF,[origo(1),origo(2),f_layer+1],1, wrist_radius,thickness);
    %%%%%%%%%%% Definerer stayOff voksler - gule
    %vGstayOff(:,:,f_layer:end) = vG_work(:,:,f_layer:end); 
    vGstayOff = zeros(size(vG_work),'int8');

    %vGstayOff = vGcylinder(vGstayOff,[origo(1),origo(2),origo(3) + round(thickness/2) + arm_height],1, wrist_radius,height);
    vGstayOff = vGcylinder(vGstayOff,[origo(1),origo(2),f_layer+1],1, wrist_radius,thickness*2);
    %vGstayOff = vGstayOff + stayoff_shell1;

    %vGextF = vGextF - vGstayOff;
    %vGextF(vGextF<0) = 0;

    vGsave = vG_work;
    amount_t1 = sum(vG_work,'all');
    x_bendp = [];
    x_bendm = [];
    y_bendp = [];
    y_bendm = [];
    
    for i = 1:gt
        fprintf("Step: %d/%d \n", i, gt)
        figure(figcount);clf;plotVg_safe(vG_work,'edgeOff');
        hold on;plotVg_safe(vGstayOff,'edgeOff','col',[0.9 0.9 0.5]);
        hold on;plotVg_safe(vGextF,'edgeOff','col',[0.5 0.5 0.5]);
        hold on;plotVg_safe(vGextC,'edgeOff','col',[0.6 0.6 0.8]);
        title("Generation: " + i + " Stay off (gul), extF (grå), låst (blå)");
        sun1;

        saveFigToAnimGif('roboarmfinal.gif', i==1);

        [E,N, ~] = vox2mesh18(vG_work);
        radius = 0.003; E(:,3) = pi*radius^2;
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
        if current_weight < 1
            if wanted_nrnodes - amount_t1 < noVoxToRemove
                vG_work = safeRemoveVoxels3D(vG_work, NstressSorted, wanted_nrnodes-amount_t1);
            end
        else
            vG_work = safeRemoveVoxels3D(vG_work, NstressSorted, noVoxToRemove);
        end
        
        for n=1:noVoxToRemove
           vG_work( NairStressSorted(n,1), NairStressSorted(n,2), NairStressSorted(n,3) ) = 1;
        end

        vG_work = gravitate2D(vG_work, vGstayOff); vG_work = gravitate2D(vG_work, vGstayOff);
        
        current_weight = PLAdens*(plainfill*nnz(vG_work)*(nozzwidth/2)^2)*pi + weight_hand; 
        
        if idx == 0
            x_bendp = [x_bendp max(dN, [], 'all')];
        elseif idx == 1
            y_bendp = [y_bendp max(dN, [], 'all')];
        elseif idx == 2
            x_bendm = [x_bendm max(dN, [], 'all')];
        else
            y_bendm = [y_bendm max(dN, [], 'all')];
        end
        figure(figcount+1);clf;t = tiledlayout(2,2); % Requires R2019b or later
        ax1 = nexttile; plot(ax1, 1:length(x_bendp), x_bendp);title(ax1,'Max bend with +x push')
        ax2 = nexttile; plot(ax2, 1:length(x_bendm), x_bendm);title(ax2,'Max bend with -x push')
        ax3 = nexttile; plot(ax3, 1:length(y_bendp), y_bendp);title(ax3,'Max bend with +y push')
        ax4 = nexttile; plot(ax4, 1:length(y_bendm), y_bendm);title(ax4,'Max bend with -y push')
        t.Padding = 'compact';
        t.TileSpacing = 'compact';

        
        amount_t1 = sum(vG_work,'all');
        fprintf('Max bent node from external forces: %4.5f\n', max(dN, [], 'all'));
        fprintf('Antall noder: %d\n', amount_t1);
        fprintf("Current weight versus wanted weight: %.5f/%.5f \n", current_weight, weight_want);
    end
    vG_work(:,:,1:b_layer-3) = vG(:,:,1:b_layer-3);
    vG = vG_work;
end