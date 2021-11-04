clear; close all;

[vG,vGc] = get_struct1();
figure(3);clf;plotVg_safe(vG, 'edgeOff');

voxSize = 1.0; % 1 mm på printern, kan forandres for bedre eller dårligere vokseloppløsning
xDist1 = round(20/voxSize); % Avstand fra venstre vegg til rød aksel. Med voxSize = 1, blir det 20stk voksler 
xDist2 = round(xDist1 + 140/voxSize); % Avstand fra rød aksel til første festeaksel
xDist3 = round(xDist2 + 40/voxSize); % Avstand fra første til siste festeaksel
radiusAxel = round(6/voxSize); % Med voxSize = 1, blir det 6stk voksler

%%%%%%%%%%% Lager initiell vG
vG = zeros(xDist1*2+xDist3, radiusAxel*20, 3, 'int8');
yCenter = round(size(vG,2)/2);
vG = vGbox(vG,[3 round(yCenter-radiusAxel*5/2) 2],1, size(vG,1)-4,radiusAxel*5,1);
xhole1 = round((xDist1+xDist2)/4);
xhole2 = round(2*(xDist1+xDist2)/4);
xhole5 = round(3*(xDist1+xDist2)/4);
vG(xhole1-3:xhole1+3, yCenter-1:yCenter+1, 2) = 0;
vG(xhole2-3:xhole2+3, yCenter-1:yCenter+1, 2) = 0;
vG(xhole5-3:xhole5+3, yCenter-1:yCenter+1, 2) = 0;


xhole3 = round((xDist2+xDist3)/2)-5;
xhole4 = xhole3 + 10;
vG(xhole3-3:xhole3+3, yCenter, 2) = 0;
%vG(xhole4-1:xhole4+1, yCenter, 2) = 0;

%%%%%%%%%%% Definerer låste voksler - blå
vGextC = zeros(size(vG),'int8');
vGextC = vGcylinder(vGextC,[xDist2,yCenter,2],1, 2,1);
vGextC = vGcylinder(vGextC,[xDist3,yCenter,2],1, 2,1);

%%%%%%%%%%% Definerer kraft voksler - grå
vGextF = zeros(size(vG),'int8');
vGextF = vGcylinder(vGextF,[xDist1,yCenter,2],1, radiusAxel,1);

%%%%%%%%%%% Definerer stayOff voksler - gule
vGstayOff = zeros(size(vG),'int8');
vGstayOff = vGcylinder(vGstayOff,[xDist1,yCenter,2],1, radiusAxel+3,1);
vGstayOff = vGcylinder(vGstayOff,[xDist2,yCenter,2],1, radiusAxel+3,1);
vGstayOff = vGcylinder(vGstayOff,[xDist3,yCenter,2],1, radiusAxel+3,1);
noVoxToRemove = 20; % for eksempel

figure(1);clf;plotVg_safe(vG, 'edgeOff');
for i=1:1000
    vG = reinforce_struct(vG, vGextC, vGextF, vGstayOff, noVoxToRemove);
    if mod(i,3) == 0
        figure(2);clf;plotVg_safe(vG, 'edgeOff');
    end
end

