clear; close all;

arm_radius = 80;
arm_height = 100;
width = arm_radius*2;
height = arm_height*4;

origo = [arm_height*2.5 arm_height*2.5 2];
thickness = 10;

% INSERT CALCULATES FOR FINDING THE LARGEST POSSIBLE WIDTH    % Use the largest possible width distance as the width of the arm 
% Placeholder for arm connection point
vG_arm = zeros(arm_height*5, arm_height*5, arm_height*5, 'int8');
vGc = vGcylinder(vG_arm, origo, 1, arm_radius, arm_height);
vGc = vGsphere(vGc, [origo(1) origo(2) arm_height], 1, arm_radius);
figure(1);clf;plotVg_safe(vGc, 'edgeOff');

vG_pro = zeros(arm_height*5, arm_height*5, arm_height*5, 'int8');
vG_pro = vGcylinder(vG_pro, origo, 1, arm_radius*2, arm_height);

vG_pro = vG_pro - vGc;

figure(1);clf;plotVg_safe(vG_pro, 'edgeOff');
