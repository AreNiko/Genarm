function singular = areAnyNeighbourNodesSingular3D(vG,meNode)
% Denne kjøres på megNoden som akkurat er fjernet, og sjekker alle
% nabo noder om de nå har fått færre enn 2 lineært uavhenginge edger til omverdenen
% som konsekvens av at megNoden er fjenet
% Har de det, kan de bevege seg og er singulære

x = meNode(1);
y = meNode(2);
z = meNode(3);

% scanner naboer
singular = 0;

for xx = x-1:x+1
    for yy = y-1:y+1
        for zz = z-1:z+1
            if vG(xx,yy,zz) == 1 % ingen vits å sjekke om et "hull" er singul�rt/løst
                if xx ~= x || yy ~= y || zz ~= z % nabo er ikke meg selv
                    if areNeighbourSingular(vG, [xx yy zz]) == 1
                        singular = 1;
                    end
                end
            end
        end
    end
end

%%
    function singular = areNeighbourSingular(vG, p)
        % Nå er "jeg" blitt p
        % Jeg er alltid 1
        
        xxx = p(1); % husk at areNeighbourSingular() deler variable med areAnyNeighbourNodeSingular()
        yyy = p(2);
        zzz = p(3);
        singular = 0;            
        
        A = vG(xxx-1:xxx+1, yyy-1:yyy+1, zzz-1:zzz+1); % min nabo matrise
        
        sumOfNeighboursToP = sum(A,'all') - 1; % tekker fra 1 siden jeg selv er 1
        
        % kun en nabonode, jeg er l�s
        if sumOfNeighboursToP < 2
            singular = 1;
            return;
        end
        
        M = zeros(2,sumOfNeighboursToP); % lager en matrise som har som rader alle edgeVektorer fra meg til nabo noder
        
        i = 1;
        for xxxx = xxx-1 : xxx+1 % scanner alle naboer
            for yyyy = yyy-1 : yyy+1 % scanner alle naboer
                for zzzz = zzz-1:zzz+1
                    if vG(xxxx,yyyy,zzzz) == 1 % funnet en nabo/selv vox
                        if xxxx ~= xxx || yyyy ~= yyy || zzzz ~= zzz % er ikke meg selv
                            M(1,i) = xxxx-xxx; % lager vektor komponent til nabo
                            M(2,i) = yyyy-yyy; % lager vektor komponent til nabo
                            M(3,i) = zzzz-zzz;
                            i = i + 1;
                        end
                    end
                end
            end
        end
        
        %M  The rank of a matrix is defined as (a) the maximum number of linearly independent column vectors
        r = rank(M);
        if r < 2
            singular = 1;
        end
    end

end
