function vG = gen_structure(height)
    vG = zeros(300, 300, 300, 'int8');
    vG = vGbox(vG, [1,1,1], 1, width, depth, height);
    vG = vGbox(vG, [10,10,1], -1, width-10, depth-10, height);
end

clear; close all;
setUp3dView;



radius = 10;
noSec = 30;
origo = [50, 50, 1];
width = 100;
depth = 100;
height = 300;

arm_radius = 50;
arm_height = 100;
vG_arm = zeros(width, depth, height, 'int8');
vGc = vGcylinder(vG_arm, origo, 1, arm_radius, arm_height);
%figure(1);clf;plotVg_safe(vGc);


figure(2);clf;plotVg_safe(vG);plotVg_safe(vGc)

%figure(3);clf;plotVg_safe(vG);plotPoly(Fc, Vc); plotPoly(Fs1, Vs1); plotPoly(Fs2, Vs2);
