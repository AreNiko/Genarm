function [vG, vGc, vGextC, vGextF, vGstayOff] = get_struct1(arm_radiusx,arm_radiusy,wrist_radius,height)

    arm_height = 14;
    thickness = 7;
    max_radius = max(arm_radiusx, arm_radiusy);
    xdim = 64;
    ydim = 55;
    % Initial enviornment
    env = zeros(xdim, ydim, 160, 'int8');
    origo = [round(xdim/2) round(ydim/2) (max_radius+thickness+10)];

    [vG, vGc, vGextC, vGextF, vGstayOff] = genstructure(env, origo, max_radius, wrist_radius, height, thickness, arm_height);
    
    %{
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
    
    env = zeros(7, 7, 7, 'int8');
    origo = [4 4 2];
    vG = env;
    vG(3:5,3:5,2:5) = 1;
    vGc = env;
    vGextC = env;
    vGextF = env;
    vGstayOff = env;
    
    vGextC(:,:,1:2) = vG(:,:,1:2); 
    vGextF(:,:,end-2:end) = vG(:,:,end-2:end);
    %}
end