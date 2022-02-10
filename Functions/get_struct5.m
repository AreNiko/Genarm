function [vG, vGextC, vGextF, vGstayOff] = get_struct5()

    voxSize = 3; % 1 mm på printern, kan forandres for bedre eller dårligere vokseloppløsning
    xDist1 = round(20/voxSize); % Avstand fra venstre vegg til rød aksel. Med voxSize = 1, blir det 20stk voksler 
    xDist2 = round(xDist1 + 40/voxSize); % Avstand fra rød aksel til første festeaksel
    xDist3 = round(xDist2 + 40/voxSize); % Avstand fra første til siste festeaksel
    radiusAxel = round(6/voxSize); % Med voxSize = 1, blir det 6stk voksler
    %%%%%%%%%%% Lager initiell vG
    vG = zeros(xDist1*2+xDist3, radiusAxel*20, 7, 'int8');
    yCenter = round(size(vG,2)/2);
    vG = vGbox(vG,[3 round(yCenter-radiusAxel*5/2) 2],1, size(vG,1)-4,radiusAxel*5,5);
    %%%%%%%%%%% Definerer låste voksler - blå
    vGextC = zeros(size(vG),'int8');
    vGextC = vGcylinder(vGextC,[xDist2,yCenter,2],1, 2,5);
    vGextC = vGcylinder(vGextC,[xDist3,yCenter,2],1, 2,5);
    %%%%%%%%%%% Definerer kraft voksler - grå
    vGextF = zeros(size(vG),'int8');
    vGextF = vGcylinder(vGextF,[xDist1,yCenter,2],1, radiusAxel,5);
    %%%%%%%%%%% Definerer stayOff voksler - gule
    vGstayOff = zeros(size(vG),'int8');
    vGstayOff = vGcylinder(vGstayOff,[xDist1,yCenter,2],1, radiusAxel+1,5);
    vGstayOff = vGcylinder(vGstayOff,[xDist2,yCenter,2],1, radiusAxel+1,5);
    vGstayOff = vGcylinder(vGstayOff,[xDist3,yCenter,2],1, radiusAxel+1,5);
    figure(1);clf;plotVg(vG,'edgeOff');
    hold on;plotVg(vGstayOff,'edgeOff','col',[0.9 0.9 0.5]);
    hold on;plotVg(vGextF,'edgeOff','col',[0.5 0.5 0.5]);
    hold on;plotVg(vGextC,'edgeOff','col',[0.6 0.6 0.8]);
    tittle("Stay off (gul), extF (grå), låst (blå)");
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