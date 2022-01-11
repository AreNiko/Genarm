function [vG, figcount] = genarm(env, origo, arm_radiusx, arm_radiusy, wrist_radius, height, thickness, gt, noVoxToRemove, weight_hand, weight_want, arm_height, addrm, withbone, figcount)
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
    
    PLAdens = 1.25; %Density of PLA in g/cm^3
    nozzwidth = 0.2; %Nozzle width of 3D printer in cm
    infill = 0.2; % Infill of 3d printer, for now set to 20%
    Voxsize = 0.4; % FOR NOW SET VOX SIZE TO 4mm / 0.4 cm
    % Area of a voxel in cm^3 with infill 
    % Infill not taking into account that the voxels are connected yet
    plainfill = (Voxsize*Voxsize*Voxsize*infill*0.001) + (Voxsize*Voxsize*6*0.01); 
    
    max_radius = max(arm_radiusx, arm_radiusy);
    [vG, vGc, ~, ~, ~] = genstructure(env, origo, max_radius, wrist_radius, height, thickness, arm_height);
    figcount=figcount+1;figure(figcount);clf;plotVg_safe(vG, 'edgeOff');
    figcount=figcount+1;figure(figcount);clf;plotVg_safe(vGc, 'edgeOff');
    
    f_layer = origo(3) + arm_height + height - thickness;
    b_layer = origo(3) + arm_height + max_radius - thickness;
    
    if withbone == 1
        [bone, figcount] = genbone(env, origo, b_layer, f_layer, max_radius, wrist_radius, height, thickness, 100, weight_hand, weight_want, figcount);
        vG = vG + bone;
    end
    vG(vG>1) = 1;
    
    current_weight = PLAdens*(plainfill*nnz(vG)*(nozzwidth/2)^2)*pi + weight_hand; 
    wanted_nrnodes = (weight_want - weight_hand)/((PLAdens*plainfill*(nozzwidth/2)^2)*pi); 
    % Add weight of biological forarm to get similar weight as an average arm?
    fprintf("Current weight of generated prosthesis versus wanted weight \n %.5f/%.5f \n", current_weight, weight_want);
    %%
    % TODO: Generalize this to work for different parameters
    vG_work = vG;
    
    %vG_work(:,:,1:b_layer-3) = 0; % Temporarily remove unnecessary part
    %vG_work(:,:,1:b_layer) = 0;
    %vG_work(:,:,f_layer:end) = 0;

    %%%%%%%%%%% Definerer låste voksler - blå
    %vGextC = zeros(size(vG_work),'int8');
    %vGextC(:,:,1:b_layer+1) = vG_work(:,:,1:b_layer+1);
    
    vGc_1more = outerShell(vGc, ceil(thickness/2)+2);
    %vGextC = vGc;
    %vGextC(vGextC>1) = 1;
    
    vGextC = vGc + vGc_1more.*vG_work;

    %%%%%%%%%%% Definerer kraft voksler - grå
    vGextF = zeros(size(vG_work),'int8');
    %vGextF(:,:,f_layer:end) = vG_work(:,:,f_layer:end);
    vGextF = vGcylinder(vGextF,[origo(1),origo(2),f_layer+1],1, wrist_radius+1,thickness);
    
    %%%%%%%%%%% Definerer stayOff voksler - gule
    %vGstayOff(:,:,f_layer:end) = vG_work(:,:,f_layer:end); 
    vGstayOff = zeros(size(vG_work),'int8');

    %vGstayOff = vGcylinder(vGstayOff,[origo(1),origo(2),origo(3) + round(thickness/2) + arm_height],1, wrist_radius,height);
    vGstayOff = vGcylinder(vGstayOff,[origo(1),origo(2),f_layer+1],1, wrist_radius+1,thickness); % Hand connection point
    %vGstayOff = vGcylinder(vGstayOff,[origo(1),origo(2),b_layer+max_radius+1],1, round(thickness/2),f_layer-(b_layer+ max_radius-thickness+1)); % Middel tube/Bone 
    %vGstayOff = vGstayOff + stayoff_shell1;
    
    %vGstayOff(:,:,1:b_layer+1) = vG_work(:,:,1:b_layer+1);
    %vGstayOff = vGstayOff + vGc_1more.*vG_work;
    
    vGstayOff(vGstayOff>1) = 1;
    %vGextF = vGextF - vGstayOff;
    %vGextF(vGextF<0) = 0;
    
    vG_work(1:origo(1), origo(2)-2:origo(2)+2, origo(3)) = 0;
    vG_work(1:origo(1), origo(2)-2:origo(2)+2, origo(3)+5) = 0;
    vG_work(1:origo(1), origo(2)-2:origo(2)+2, origo(3)-5) = 0;
    vG_work(1:origo(1), origo(2)-14:origo(2)-9, origo(3)) = 0;
    vG_work(1:origo(1), origo(2)+9:origo(2)+14, origo(3)) = 0;
    
    figcount=figcount+1;figure(figcount);clf;plotVg_safe(vG_work(:,origo(2):end,:), 'edgeOff');
    %[N, E, F, C, vGno] = make_N_E_F_C_vGno(vG, 2, 1);
    %[sE, dN] = FEM_truss(N,E, F,C);
    %figcount=figcount+1;figure(figcount);clf;plotMesh(N,E); % evt. plotMesh(N,E,'txt');

    figcount=figcount+1;
    
    vGsave = vG_work;
    amount_t1 = sum(vG_work,'all');
    x_bendp = [];
    xy_bendp = [];
    y_bendp = [];
    xy_bend = [];
    x_bendm = [];
    xy_bendm = [];
    y_bendm = [];
    yx_bend = [];
    % Generative steps
    for i = 1:gt
        fprintf("Step: %d/%d \n", i, gt)
        
        vGextF(1:origo(1), origo(2)-1:origo(2)+1, 1:b_layer) = vG_work(1:origo(1), origo(2)-1:origo(2)+1, 1:b_layer);
        vGextF(origo(1)-1:origo(1)+1, 1:origo(2), 1:b_layer) = vG_work(origo(1)-1:origo(1)+1, 1:origo(2), 1:b_layer);
        vGextF(origo(1):end, origo(2)-1:origo(2)+1, 1:b_layer) = vG_work(origo(1):end, origo(2)-1:origo(2)+1, 1:b_layer);
        vGextF(origo(1)-1:origo(1)+1, origo(2):end, 1:b_layer) = vG_work(origo(1)-1:origo(1)+1, origo(2):end, 1:b_layer);
        vGextF = vGextF - vGc;
        vGextF(vGextF<0) = 0;
        
        figure(figcount);clf;plotVg_safe(vG_work,'edgeOff');
        hold on;plotVg_safe(vGstayOff,'edgeOff','col',[0.9 0.9 0.5]);
        hold on;plotVg_safe(vGextF,'edgeOff','col',[0.5 0.5 0.5]);
        hold on;plotVg_safe(vGextC,'edgeOff','col',[0.6 0.6 0.8]);
        title("Generation: " + i + " Stay off (gul), extF (grå), låst (blå)");
        sun1;

        saveFigToAnimGif('roboarm-strengthen.gif', i==1);
        
        figure(figcount+1);clf;plotVg_safe(vG_work(:,origo(2):end,:));
        hold on;plotVg_safe(vGstayOff(:,origo(2):end,:),'col',[0.9 0.9 0.5]);
        hold on;plotVg_safe(vGextF(:,origo(2):end,:),'col',[0.5 0.5 0.5]);
        hold on;plotVg_safe(vGextC(:,origo(2):end,:),'col',[0.6 0.6 0.8]);
        title("Generation: " + i + " Dissection");
        sun1;
        saveFigToAnimGif('roboarm-half-strengthen.gif', i==1);

        [E,N, ~] = vox2mesh18(vG_work);
        radius = 0.003; E(:,3) = pi*radius^2;
        %fprintf("%d %d \n", size(E), size(N))
        extC = vGextC2extC(vGextC,vG_work);
        extF = vGextF2extF(vGextF,vG_work);
        
        idx = mod(i,8);
        if idx == 0
            extF = extF.* [100 0 0];
        elseif idx == 1
            extF = extF.* [50 50 0];
        elseif idx == 2
            extF = extF.* [0 100 0];
        elseif idx == 3
            extF = extF.* [-50 50 0];
        elseif idx == 4
            extF = extF.* [-100 0 0];
        elseif idx == 5
            extF = extF.* [-50 -50 0];
        elseif idx == 6
            extF = extF.* [0 -100 0];
        elseif idx == 7
            extF = extF.* [50 -50 0];
        end
        
        extC(:,3) = 0;
        %{
        figure(figcount+1);clf;plotMesh(N,E,'lTykk',0.01);
        hold on;plotMeshCext(N,extC,1);
        hold on;plotMeshFext(N,extF, 0.01);
        %}
        [sE, dN] = FEM_truss2(N,E, extF,extC); % The most taxing process 
        
        %visuellScale = 100;
        %figure(figcount+1);clf;plotMesh(N-visuellScale*dN,E,'lTykk',0.01); % Mesh med nodeforflytninger

        Nstress  = edgeStress2nodeStress(N,E, sE);
        NairStress  = nodeStress2airNodeStress(Nstress,vG_work);
        Nstress = cutStayOffNodes(Nstress, vGstayOff);
        NairStress = cutStayOffNodes(NairStress, vGstayOff);
        NstressSorted = sortrows(Nstress,4,'ascend');
        NairStressSorted = sortrows(NairStress,4,'descend');
        if current_weight < weight_want
            if addrm == 1
                if wanted_nrnodes - amount_t1 < noVoxToRemove
                    vG_work = safeRemoveVoxels3D(vG_work, NstressSorted, wanted_nrnodes-amount_t1);
                end
            else
                [vG_work, reduce] = safeRemoveVoxels3D(vG_work, NstressSorted, noVoxToRemove);
            end
        else
             [vG_work, reduce] = safeRemoveVoxels3D(vG_work, NstressSorted, noVoxToRemove);
        end
  
        for n=1:reduce
           vG_work( NairStressSorted(n,1), NairStressSorted(n,2), NairStressSorted(n,3) ) = 1;
        end

        vG_work = gravitate2D(vG_work, vGstayOff); vG_work = gravitate2D(vG_work, vGstayOff);
        
        current_weight = PLAdens*(plainfill*(nnz(vG_work)+nnz(vG(:,:,1:b_layer-3)) - nnz(vGc))*(nozzwidth/2)^2)*pi + weight_hand; 
        
        if mod(i,20) == 0
            vGsave = vG_work;
        end
        
        if idx == 0
            x_bendp = [x_bendp max(sum(abs(dN),2), [], 'all')];
        elseif idx == 1
            xy_bendp = [xy_bendp max(sum(abs(dN),2), [], 'all')];
        elseif idx == 2
            y_bendp = [y_bendp max(sum(abs(dN),2), [], 'all')];
        elseif idx == 3
            xy_bend = [xy_bend max(sum(abs(dN),2), [], 'all')];
        elseif idx == 4
            x_bendm = [x_bendm max(sum(abs(dN),2), [], 'all')];
        elseif idx == 5
            xy_bendm = [xy_bendm max(sum(abs(dN),2), [], 'all')];
        elseif idx == 6
            y_bendm = [y_bendm max(sum(abs(dN),2), [], 'all')];
        elseif idx == 7
            yx_bend = [yx_bend max(sum(abs(dN),2), [], 'all')];
        end
        figure(figcount+2);clf;t = tiledlayout(2,4); % Requires R2019b or later
        title("Generation: " + i + " Bending");
        ax1 = nexttile; plot(ax1, 1:length(x_bendp), x_bendp);title(ax1,'Max bend with +x push');
        ax2 = nexttile; plot(ax2, 1:length(xy_bendp), xy_bendp);title(ax2,'Max bend with +x +y push')
        ax3 = nexttile; plot(ax3, 1:length(y_bendp), y_bendp);title(ax3,'Max bend with +y push')
        ax4 = nexttile; plot(ax4, 1:length(xy_bend), xy_bend);title(ax4,'Max bend with -x +y push')
        ax5 = nexttile; plot(ax5, 1:length(x_bendm), x_bendm);title(ax5,'Max bend with -x push')
        ax6 = nexttile; plot(ax6, 1:length(xy_bendm), xy_bendm);title(ax6,'Max bend with -x -y push')
        ax7 = nexttile; plot(ax7, 1:length(y_bendm), y_bendm);title(ax7,'Max bend with -y push')
        ax8 = nexttile; plot(ax8, 1:length(yx_bend), yx_bend);title(ax8,'Max bend with +x -y push')
        %t.Padding = 'compact';
        %t.TileSpacing = 'compact';

        islands = bwconncomp(vG_work).NumObjects;
        fprintf('Number of isolated structures: %d\n', islands);
        amount_t1 = sum(vG_work,'all');
        fprintf('Max bent node from external forces: %4.5f\n', max(abs(dN), [], 'all'));
        fprintf('Antall noder: %d\n', amount_t1);
        fprintf("Current weight versus wanted weight: %.5f/%.5f \n", current_weight, weight_want);
    end
    %vG_work(:,:,1:b_layer-3) = vG(:,:,1:b_layer-3);
    vG_work = vG_work + vGextC;
    
    vG_work(vG_work>1) = 1;
    if thickness > 4
        vG_work = outerFillet(vG_work,thickness);
    end
    figcount=figcount+3;figure(figcount);clf;plotVg_safe(vG_work, 'edgeOff');
    sun1;
    vG = vG_work;
end