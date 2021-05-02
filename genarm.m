
clear; close all;
setUp3dView;
figcount = 0;
% inputs
% Generative steps
gt = 500;
% Dimension
arm_radiusx = 20;
arm_radiusy = 20;
max_radius = max(arm_radiusx, arm_radiusy);
wrist_radius = 15;
arm_height = 30;
thickness = 5;
width = (arm_radiusx+thickness)*2;
depth = (arm_radiusy+thickness)*2;
height = 150;

% Weight
weight_hand = 0.58;
weight_want = 1.00;

% Initial enviornment
env = zeros(height+max_radius, height+max_radius, height*2, 'int8');

origo = [round(height/2) round(height/2) max_radius+thickness+10];
vG = vGcone(env, [origo(1) origo(2) origo(3)+arm_height], width, depth, height, wrist_radius);
figcount=figcount+1;figure(figcount);clf;plotVg_safe(vG, 'edgeOff');
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
%%
% TODO: Generalize this to work for different parameters
vG_work = vG;
f_layer = origo(3) + arm_height + height - thickness;
b_layer = origo(3) + arm_height + max_radius - thickness;

%vG_work(:,:,1:b_layer) = 0;
%vG_work(:,:,f_layer:end) = 0;

%%%%%%%%%%% Definerer låste voksler - blå
vGextC = zeros(size(vG_work),'int8');
vGextC(:,:,1:b_layer+1) = vG_work(:,:,1:b_layer+1); 


%%%%%%%%%%% Definerer kraft voksler - grå
vGextF = zeros(size(vG_work),'int8');
vGextF(:,:,f_layer:end) = vG_work(:,:,f_layer:end);

%%%%%%%%%%% Definerer stayOff voksler - gule
%vGstayOff = zeros(size(vG_work),'int8');
%vGstayOff(:,:,f_layer:end) = vG_work(:,:,f_layer:end); 
vGstayOff = zeros(size(vG_work),'int8');
vGstayOff = vGcylinder(vGstayOff,[origo(1),origo(2),f_layer],1, wrist_radius,6);

vGextF = vGextF - vGstayOff;
noVoxToRemove = 200; % for eksempel
vGsave = vG_work;

for i = 1:gt
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
    [sE, dN] = FEM_truss(N,E, extF,extC);
    
    visuellScale = 100;
    %figure(figcount+1);clf;plotMesh(N-visuellScale*dN,E,'lTykk',0.01); % Mesh med nodeforflytninger
    
    Nstress  = edgeStress2nodeStress(N,E, sE);
    NairStress  = nodeStress2airNodeStress(Nstress,vG_work);
    Nstress = cutStayOffNodes(Nstress, vGstayOff);
    NairStress = cutStayOffNodes(NairStress, vGstayOff);
    NstressSorted = sortrows(Nstress,4,'ascend');
    NairStressSorted = sortrows(NairStress,4,'descend');
    
    try
       vG_work = safeRemoveVoxels2D(vG_work, NstressSorted, noVoxToRemove);
       for n=1:noVoxToRemove
           vG_work( NairStressSorted(n,1), NairStressSorted(n,2), NairStressSorted(n,3) ) = 1;
       end
       vG_work = gravitate2D(vG_work, vGstayOff); vG_work = gravitate2D(vG_work, vGstayOff);
       if mod(i,20) == 0
           vGsave = vG_work;
       end
    catch exception
       if noVoxToRemove > 5
           noVoxToRemove = noVoxToRemove - 5;
       else
           noVoxToRemove = round(noVoxToRemove/2);
       end
       vG_work = vGsave;
    end
end
figcount = figcount+1;
figcount = figcount+1;figure(figcount);clf;plotVg_safe(vG_work,'edgeOff');
hold on;plotVg_safe(vGstayOff,'edgeOff','col',[0.9 0.9 0.5]);
hold on;plotVg_safe(vGextF,'edgeOff','col',[0.5 0.5 0.5]);
hold on;plotVg_safe(vGextC,'edgeOff','col',[0.6 0.6 0.8]);
title("Stay off (gul), extF (grå), låst (blå)");
sun1;


%figure(figcount);clf;plotVg_safe(vG_work,'edgeOff');
%% 
if thickness > 4
    vG_work1 = outerFillet(vG_work,thickness);
end
figcount=figcount+1;figure(figcount);clf;plotVg_safe(vG_work1, 'edgeOff');


[F,V] = vox2quadPoly(vG_work);
[fTri]= quadPoly2triPoly(F);
%figure(5);clf;plotPoly(fTri,V);
%triPoly2stl(fTri,V, 'test_smooth.stl', 5);

vG_half1 = vG_work;
vG_half1(:,origo(2)+1:end,:) = 0;
figcount=figcount+1;figure(figcount);clf;plotVg_safe(vG_half1, 'edgeOff');
[F1,V1] = vox2quadPoly(vG_half1);
[V1] = smoothQuadVertex(F1,V1);
[V1] = smoothQuadVertex(F1,V1);
[V1] = smoothQuadVertex(F1,V1);
[fTri1]= quadPoly2triPoly(F1);
figcount=figcount+1;figure(figcount);clf;plotPoly(fTri1,V1);

vG_half2 = vG_work1;
vG_half2(:,1:origo(2)-1,:) = 0;
figcount=figcount+1;figure(figcount);clf;plotVg_safe(vG_half2, 'edgeOff');
[F2,V2] = vox2quadPoly(vG_half2);
[V2] = smoothQuadVertex(F2,V2);
[V2] = smoothQuadVertex(F2,V2);
[V2] = smoothQuadVertex(F2,V2);
[fTri2]= quadPoly2triPoly(F2);
figcount=figcount+1;figure(figcount);clf;plotPoly(fTri2,V2);

triPoly2stl(fTri1,V1, 'test3_armhalf1.stl', 0.1)
triPoly2stl(fTri2,V2, 'test3_armhalf2.stl', 0.1)
