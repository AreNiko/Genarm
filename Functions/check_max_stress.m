function [max_stress, max_bend, comps] = check_max_stress(vG, vGextC, vGextF)
    [E,N, ~] = vox2mesh18(vG);
    radius = 0.003; E(:,3) = pi*radius^2;

    extC = vGextC2extC(vGextC,vG);
    extF = vGextF2extF(vGextF,vG);

    extF = extF.* [0 10 0];
    extC(:,3) = 0;
    
    [sE, dN] = FEM_truss(N, E, extF, extC);
    max_bend = max(sum(abs(dN),2), [], 'all');
    max_stress = max(sum(abs(sE),2), [], 'all');
    
    comps = bwconncomp(vG, 6).NumObjects;