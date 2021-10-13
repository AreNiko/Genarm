function [vG, vGc] = get_struct1()
    arm_radiusx = 16;
    arm_radiusy = 16;

    wrist_radius = 12; % Radius of my arm is about 30 mm 
    arm_height = 15;
    thickness = 7;

    height = 100; % Average forearm length is 46 cm, so 460 mm

    max_radius = max(arm_radiusx, arm_radiusy);

    % Initial enviornment
    env = zeros(height, height, height*2, 'int8');
    origo = [round(height/2) round(height/2) max_radius+thickness+10];

    [vG, vGc] = genstructure(env, origo, max_radius, wrist_radius, height, thickness, arm_height);

    %save('vG.mat','vG', '-ascii');
    %C = permute(vG,[1 3 2]);
    %C = reshape(vG,[],size(vG,2),1);
    %writematrix(C,'vG.txt')
end