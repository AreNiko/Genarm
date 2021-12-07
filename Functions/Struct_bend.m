function [sE, dN] = Struct_bend(vG, vGextC, vGextF)
    tic
    [E,N, ~] = vox2mesh18(vG);
    radius = 0.003; E(:,3) = pi*radius^2;
    
    extC = vGextC2extC(vGextC,vG);
    extF = vGextF2extF(vGextF,vG);

    extF = extF.* [0 10 0];

    extC(:,3) = 0;

    [sE, dN] = FEM_truss(N,E, extF,extC); % The most taxing process 
end