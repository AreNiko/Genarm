function [vG, reduce] = safeRemoveVoxels3D(vG, NstressSorted, noVoxToRemove)
% Her gjør man noen sjekker før man fjerner vox
% Disse sjekkene er ikke nok til å 100% eliminere sing, må filtrere beam til slutt

noRemoved = 0;
reduce = noVoxToRemove;
n = 1;
while noRemoved < noVoxToRemove && n < size(NstressSorted,1)
    proposedVoxel = NstressSorted(n,1:4);
    xP = proposedVoxel(1);
    yP = proposedVoxel(2);
    zP = proposedVoxel(3);
    
    % Fjerner vox, men vi kan ombestemme oss
    vG(xP,yP,zP) = 0; 
    noRemoved = noRemoved + 1;
    neigh = vG(xP-1,yP,zP)+vG(xP,yP-1,zP)+vG(xP,yP,zP-1)+vG(xP+1,yP,zP)+vG(xP,yP+1,zP)+vG(xP,yP,zP+1);
    %neigh = sum(vG(xP-1:xP+1, yP-1:yP+1, zP-1:zP+1),'all');
    % test nr. 1 - er proposedVoxel ikke i kontakt med luft - Vi vil ikke lage nye hull
    %if neigh == 6 % alle nabo voksler er tilstede, vi er inne i beam
    %    vG(xP,yP,zP) = 1; % Ombestemmer oss og setter proposedVoxel tilbake
    %    noRemoved = noRemoved-1;
    %end
    
    % test nr. 2 - sjekker alle nabonodene for å se om noen er løse
    if areAnyNeighbourNodesSingular3D(vG,proposedVoxel) == 1 % en eller flere nabo voksler er nå løse
        vG(xP,yP,zP) = 1; % Ombestemmer oss og setter proposedVoxel tilbake
        noRemoved = noRemoved-1;
    end

    %{
    if areAnyNeighbourNodesSingular3D(vG,proposedVoxel) == 1 % en eller flere nabo voksler er nå løse
        vG(xP,yP,zP) = 1; % Ombestemmer oss og setter proposedVoxel tilbake
        noRemoved = noRemoved-1;
    end    
    %}
    
    n = n+1;
end
if bwconncomp(vG).NumObjects > 1
    [vG, reduce] = safeRemoveVoxels3D(vG, NstressSorted, round(noVoxToRemove/2)); % Ombestemmer oss og setter proposedVoxel tilbake
end
end