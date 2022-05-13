clear; close all;

[vG, ~, vGextC, vGextF, vGstayOff] = get_struct1(14,14,12,100);
%[vG, vGc, vGextC, vGextF, vGstayOff] = get_struct3();
%[vG, vGc, vGextC, vGextF, vGstayOff] = get_struct4();
%[vG, vGextC, vGextF, vGstayOff] = get_struct5();
%vGextC(:,:,3:end) = 0;
comps = bwconncomp(vG, 6).NumObjects;
disp(comps);
figure(1);clf;plotVg_safe(vG,'edgeOff');
hold on;plotVg_safe(vGstayOff,'edgeOff','col',[0.9 0.9 0.5]);
hold on;plotVg_safe(vGextF,'edgeOff','col',[0.5 0.5 0.5]);
hold on;plotVg_safe(vGextC,'edgeOff','col',[0.6 0.6 0.8]);
title("Stay off (gul), extF (grå), låst (blå)");
sun1;

for i=1:1
    [vG, sE, dN] = reinforce_struct(vG, vGextC, vGextF, vGstayOff, 10);
    fprintf('Max bent node from external forces: %4.5f\n', max(abs(dN), [], 'all'));
end