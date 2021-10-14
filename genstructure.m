function [vG, vGc, vGextC, vGextF, vGstayOff] = genstructure(env, origo, max_radius, wrist_radius, height, thickness, arm_height)
    
    width = (max_radius+thickness)*2;
    depth = (max_radius+thickness)*2;

    vG = vGcone(env, [origo(1) origo(2) origo(3)+arm_height], width, depth, height, wrist_radius);
    %figcount=figcount+1;figure(figcount);clf;plotVg_safe(vG, 'edgeOff');
    %stayoff_shell1 = outerShell(vG, 1);
    %figure(2);clf;plotMesh(N,E);
    % INSERT CALCULATES FOR FINDING THE LARGEST POSSIBLE radius    % Use the largest possible width distance as the radius of the arm 
    % Placeholder for arm connection point

    vGc = env;
    for i = 1:arm_height
        vGc = vGsphere(vGc, [origo(1) origo(2) origo(3)+i], 1, max_radius);
        vGc = vGsphere(vGc, [origo(1)+i origo(2) origo(3)], 1, max_radius);
    end

    vGc = innerFillet(vGc,10);
    %figcount=figcount+1;figure(figcount);clf;plotVg_safe(vGc, 'edgeOff');
    
    % Creates shell around arm and adds to structure
    vGc_shell = outerShell(vGc, thickness);
    vG = vG + vGc_shell;
    vG(vG>1) = 1;
    
    % Hollows main structure and creates connection point
    vG = innerShell(vG,thickness) - vGc;
    vGc1 = shift_3dmatrix(vGc, round((thickness+max_radius)/4),0,0);

    vG = vG - vGc1;
    vG(vG<0) = 0;
    
    vG(origo(1) + (max_radius+thickness):end,:,:) = 0;
    
    vGc_1more = outerShell(vGc, ceil(thickness/2)+2);
    
    f_layer = origo(3) + arm_height + height - thickness;
    b_layer = origo(3) + arm_height + max_radius - thickness;
    
    vGextC = vGc + vGc_1more.*vG;

    %%%%%%%%%%% Definerer kraft voksler - grå
    vGextF = zeros(size(vG),'int8');
    vGextF = vGcylinder(vGextF,[origo(1),origo(2),f_layer+1],1, wrist_radius+1,thickness);
    
    %%%%%%%%%%% Definerer stayOff voksler - gule
    vGstayOff = zeros(size(vG),'int8');
    vGstayOff = vGcylinder(vGstayOff,[origo(1),origo(2),f_layer+1],1, wrist_radius+1,thickness); % Hand connection point
    
    vGstayOff(vGstayOff>1) = 1;
end