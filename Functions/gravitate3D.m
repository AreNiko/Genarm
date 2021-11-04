function vGut = gravitate3D(vG, vGstayOff)
% Filtrerer uten å endre antall voksler
% Krever margin 2 eller større
% Flytter alle vox et hakk hvis den nye pos gir flere naboer
% Kjører en iterasjon over hele vG, men har kun lov å flytte på noder i
% Nstress_sortedAndCut slik at beskyttede noder blir unng�tt

[dimX,dimY,dimZ] = size(vG);
vGut = vG;

for x = 2 : dimX - 1 % itererer alle pos i vG
    for y = 2 : dimY - 1 % itererer alle pos i vG
        for z = 2 : dimZ - 1
            if vG(x,y,z) == 1
                if vGstayOff(x,y,z) == 0

                    % flytter orginal vox inn i alle 0 nabo pos inkludert original pos for � finne nabo sum
                    squareSumMax = 0;
                    bestX = 0;
                    bestY = 0;
                    bestZ = 0;
                    vGut(x,y,z) = 0; % nuller ut orginal pos, orginal pos konkurerer på lik linje med all 0 type naboer
                    for xx = x-1 : x+1
                        for yy = y-1 : y+1
                            for zz = z-1:z+1
                                if vGstayOff(xx,yy,zz) == 0
                                    if vGut(xx,yy,zz) == 0 % finner en ledig nabo 0 pos, eller orginal 0 pos
                                        squareSum = sum(vGut(xx-1:xx+1, yy-1:yy+1, zz-1:zz+1),'all'); % sum av alle naboer og orginal som er 0
                                        if squareSum > squareSumMax
                                            bestX = xx;
                                            bestY = yy;
                                            bestZ = zz;
                                            squareSumMax = squareSum;
                                        end
                                    end
                                end
                            end

                        end
                    end
                    vGut(bestX,bestY,bestZ) = 1;

                end
            end
        end
    end
end



end