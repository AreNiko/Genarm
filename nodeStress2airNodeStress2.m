function NairStress  = nodeStress2airNodeStress(Nstress,vG)
% Node array inneholder: kun noder i luft som er nabo til kropp, 
% max abs eller sum av stress fra alle nabo NODER
% format:
% X Y Z stress
% X Y Z stress
% X Y Z stress

[dimX, dimY, dimZ] = size(vG);
NairStress = zeros(dimX*dimY*dimZ-nnz(vG),4); % overdimensjonert

% lager hjelpe grid
vGnodeStress = zeros(dimX, dimY, dimZ);
for n = 1 : size(Nstress,1)
    % itererer alle edger i E og finner 2 noder per edge
    vGnodeStress(Nstress(n,1), Nstress(n,2), Nstress(n,3)) = Nstress(n,4);
end


i = 0;
for z = 2 : dimZ-1
    for x = 2 : dimX-1
        for y = 2 : dimY-1
            if vG(x,y,z) == 0 % er i luft
                
                %stressMax = max(max(max( vGnodeStress(x-1:x+1, y-1:y+1, z-1:z+1) ))); % Max
                %stressMax = sum(vGnodeStress(x-1:x+1, y-1:y+1, z-1:z+1),'all'); % Sum
                stressMax = vGnodeStress(z-1,y,z)+vGnodeStress(x,y-1,z)+vGnodeStress(x,y,z-1)+vGnodeStress(x+1,y,z)+vGnodeStress(x,y+1,z)+vGnodeStress(x,y,z+1);
                if stressMax>0 % Er nabo til kroppen
                    i = i + 1;
                    NairStress(i,1) = x;
                    NairStress(i,2) = y;
                    NairStress(i,3) = z;
                    NairStress(i,4) = stressMax;
                end
                
            end
        end
    end
end

% kutter vekk ubrukte posisjoner
NairStress = NairStress(1:i,:);


end