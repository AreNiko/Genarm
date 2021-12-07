clear; close all;
setUp3dView;
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
% Dimension

arm_radiusx = 18;
arm_radiusy = 18;

wrist_radius = 14; % Radius of my arm is about 30 mm 
arm_height = 17;
thickness = 9;

height = 100; % Average forearm length is 46 cm, so 460 mm

max_radius = max(arm_radiusx, arm_radiusy);

% Initial enviornment
env = zeros(height, height, height*2, 'int8');
origo = [round(height/2) round(height/2) max_radius+thickness+10];

% Weight

weight_hand = 0.58*1000; % 0.58 kg of hand converted to grams 
weight_want = 1.87*1000/3.1; % Average weight of an the forearm 1.87 kg converted to grams
figcount = 0;
gt = 1000;
noVoxToRemove = 100;
addrm = 0;
[vG_work1, figcount] = genarm(env, origo, arm_radiusx, arm_radiusy, wrist_radius, height, thickness, gt, noVoxToRemove, weight_hand, weight_want, arm_height, addrm, 0, figcount);

%% 

[F,V] = vox2quadPoly(vG_work1);
[fTri]= quadPoly2triPoly(F);
%figure(5);clf;plotPoly(fTri,V);
%triPoly2stl(fTri,V, 'test_smooth.stl', 5);

vG_half1 = vG_work1;
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