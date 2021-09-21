clear; close all;
figcount = 0;
% Dimension
arm_radiusx = 16;
arm_radiusy = 8;

wrist_radius = 12; % Radius of my arm is about 30 mm 
arm_height = 15;
thickness = 7;

height = 100; % Average forearm length is 46 cm, so 460 mm

max_radius = max(arm_radiusx, arm_radiusy);

% Initial enviornment
env = zeros(max_radius*8, max_radius*8, height*2, 'int8');
origo = [round(height/2) round(height/2) max_radius+thickness+10];

vGc = env;
for i = 1:arm_height
    vGc = vGsphere(vGc, [origo(1) origo(2) origo(3)+i], 1, max_radius);
    vGc = vGsphere(vGc, [origo(1)+i origo(2) origo(3)], 1, max_radius);
end

%vGc = innerFillet(vGc,1);
% Creates shell around arm and adds to structure
vGc_shell = outerShell(vGc, thickness);

vGc = vGc + vGc_shell;
vGc(vGc>1) = 0;

figcount=figcount+1;figure(figcount);clf;plotVg_safe(vGc, 'edgeOff');

dim_size = size(env);
center = [0 0];
center_list = [];
begin = 0;
for k=1:dim_size(3)
    if sum(vGc(:,:,k), 'all') > 0
        center = [0 0];
        begin = 1;
    end
    for i=1:dim_size(1)
        for j=1:dim_size(2)
            if vGc(i,j,k) == 1
                center(1) = center(1) + i;
                center(2) = center(2) + j;
            end
        end
    end
    if sum(vGc(:,:,k), 'all') > 0
        center = center/sum(vGc(:,:,k), 'all');
    end
    if begin == 1
        center_list = [center_list; center k];
    end
end

figcount=figcount+1;figure(figcount);clf;scatter3(center_list(:,1),center_list(:,2),center_list(:,3),'filled');
axis([1, dim_size(1), 1, dim_size(2), 1, dim_size(3)])