function [vG2, vGc, vGextC2, vGextF2, vGstayOff2] = get_struct4()

    arm_height = 10;
    thickness = 6;
    wrist_radius = 6;
    max_radius = 5;
    height = 30;
    xdim = 50;
    ydim = 50;
    % Initial enviornment
    env = zeros(xdim, ydim, 100, 'int8');
    origo = [round(xdim/2) round(ydim/2) (max_radius+thickness+10)];

    [vG, vGc, vGextC, vGextF, vGstayOff] = genstructure(env, origo, max_radius, wrist_radius, height, thickness, arm_height);
    vGextC = vGextC - vGc;
    cutz = max_radius*2 + thickness + arm_height+5;
    cutxy = 15;
    endz = 40;
    vG2 = vG(cutxy:end-cutxy,cutxy:end-cutxy,cutz:end-endz); 
    vGstayOff2 = vGstayOff(cutxy:end-cutxy,cutxy:end-cutxy,cutz:end-endz); 
    vGextF2 = vGextF(cutxy:end-cutxy,cutxy:end-cutxy,cutz:end-endz); 
    vGextC2 = vGextC(cutxy:end-cutxy,cutxy:end-cutxy,cutz:end-endz); 
    vGextC2(:,:,1:2) = vG2(:,:,1:2);
    
    figure(1);clf;plotVg(vG2,'edgeOff');
    hold on;plotVg(vGstayOff2,'edgeOff','col',[0.9 0.9 0.5]);
    hold on;plotVg(vGextF2,'edgeOff','col',[0.5 0.5 0.5]);
    hold on;plotVg(vGextC2,'edgeOff','col',[0.6 0.6 0.8]);
    title("Stay off (gul), extF (grå), låst (blå)");
    sun1;
    %{
    max_bend = check_max_bend(vG, vGextC, vGextF);
    disp(max_bend);
    
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