function [vG, vGextC, vGextF, vGstayOff] = get_struct2()
    %{
    arm_height = 14;
    thickness = 7;
    max_radius = max(arm_radiusx, arm_radiusy);
    xdim = 64;
    ydim = 55;
    % Initial enviornment
    env = zeros(xdim, ydim, 160, 'int8');
    origo = [round(xdim/2) round(ydim/2) (max_radius+thickness+10)];

    [vG, vGc, vGextC, vGextF, vGstayOff] = genstructure(env, origo, max_radius, wrist_radius, height, thickness, arm_height);
    
    
    arm_radiusx = 14;
    arm_radiusy = 14;

    wrist_radius = 12; % Radius of my arm is about 30 mm 
    arm_height = 14;
    thickness = 7;

    height = 100; % Average forearm length is 46 cm, so 460 mm

    max_radius = max(arm_radiusx, arm_radiusy);
    xdim = (max_radius+thickness+arm_height+10)*2;
    ydim = (max_radius+thickness+10)*2;
    zdim = height+(max_radius+thickness+10)*2;
    % Initial enviornment
    env = zeros(xdim, ydim, zdim, 'int8');
    origo = [round(ydim/2) round(ydim/2) max_radius+thickness+10];

    [vG, vGc, vGextC, vGextF, vGstayOff] = genstructure(env, origo, max_radius, wrist_radius, height, thickness, arm_height);
    
    %save('vG.mat','vG', '-ascii');
    %C = permute(vG,[1 3 2]);
    %C = reshape(vG,[],size(vG,2),1);
    %writematrix(C,'vG.txt')
    
    arm_radiusx = 16;
    arm_radiusy = 16;

    wrist_radius = 12; % Radius of my arm is about 30 mm 
    arm_height = 8;
    thickness = 9;

    height = 50; % Average forearm length is 46 cm, so 460 mm

    max_radius = max(arm_radiusx, arm_radiusy);

    % Initial enviornment
    env = zeros(height, height, height*2, 'int8');
    origo = [round(height/2) round(height/2) max_radius];
    
    vG = vGcone(env, origo, max_radius, max_radius, height, wrist_radius);
    vGc = env;
    vGextC = env;
    vGextF = env;
    vGstayOff = env;
    vGextC(:,:,1:max_radius+2) = vG(:,:,1:max_radius+2);
    vGextF(:,:,height+14:end) = vG(:,:,height+14:end);
    %}
    env = zeros(10, 10, 10, 'int8');
    vG = env;
    vG(4:6,4:6,2:6) = 1;
    vGextC = env;
    vGextF = env;
    vGstayOff = env;
    
    vGstayOff(:,:,4) = vG(:,:,4); 
    vGextC(:,:,1:2) = vG(:,:,1:2); 
    vGextF(:,:,end-4:end) = vG(:,:,end-4:end);
    
    %figure(1);clf;plotVg_safe(vG,'edgeOff');
    %hold on;plotVg_safe(vGstayOff,'edgeOff','col',[0.9 0.9 0.5]);
    %hold on;plotVg_safe(vGextF,'edgeOff','col',[0.5 0.5 0.5]);
    %hold on;plotVg_safe(vGextC,'edgeOff','col',[0.6 0.6 0.8]);
end