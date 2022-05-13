    % Copyright Mats H�vin, denne er optimalisert (mye jobb), please ikke del eller vis
% noen andre p� nettet (vi trenger et pub forsprang)

function [eS,dU] = FEM_truss2(N, E, extF,extC, varargin)
% FEM/FEA, The Direct Stiffness Method / Matrix Stiffness Method
% N = nodeliste (n,3), E = koblingsliste (e,3) edges der siste kolonne er areal
% eS = edgestress (e,1), dU = array av node displacement vektorer (n,3)
% extF er ext forces (n,3), extC er l�ste noder (n,3) - [0 0 0] er l�st i XYZ retning
% Har disablet GPU da det g�r mye raskere p� denne m�ten med mange kjerner,
% man kan lett kj�re i GPU hvis behov, kontakt Mats
% Alle E dim er i meter, krefter i extF er i Newton
% 'verb',val - vorboose
% 'eMod',val - elasticity modulus, default = 2e9;

verbose = 0;
EModulus = 2e9;
readVgrarginInput(); % se nederst

A = E(:,3)'; % Areal p� stag tverrsnitt, user spesifisert

extC = reshape(extC', [], 1);
extF = reshape(extF', [], 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% K
numberOfEdges=size(E,1);
numberOfNodes=size(N,1);
kDof=3*numberOfNodes;

if verbose
    fprintf('FEM Truss Direct-Stiffness-Method:\n%d noder, %d edger --- K lages ... ', numberOfNodes, numberOfEdges);
end

dim = numberOfEdges*9*2 + numberOfNodes*3 + (numberOfNodes*3-1)*2; % antall 3X3 clash + 3 diagonal linjer, vil bli redusert

I = zeros(dim, 1); % X, ned
J = zeros(dim, 1); % y, bort
X = zeros(dim, 1);

IJindexTeller = 0;
for e=1:numberOfEdges % runner alle edger/element
    indice = E(e,:); % 2 ende nodenummer [n1 n2] p� edge
    % genererer indekser i K for 6x6 matrise basert p� edge endepunkt x1 y1 z1  x2 y2 z2, n1 og n2 er ikke n�dv naboer i K
    edgeDof=[3*indice(1)-2 3*indice(1)-1 3*indice(1)  3*indice(2)-2 3*indice(2)-1 3*indice(2)]; % [3a-2 3a+1 3a  3b-2 3b-1 3b]
    x1=N(indice(1),1); % venstre node nr i edge sin x verdi
    y1=N(indice(1),2); % venstre node nr i edge sin y verdi
    z1=N(indice(1),3); % venstre node nr i edge sin z verdi
    x2=N(indice(2),1); % h�yre node nr i edge sin x verdi
    y2=N(indice(2),2); % h�yre node nr i edge sin y verdi
    z2=N(indice(2),3); % h�yre node nr i edge sin z verdi
    L = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1)); % lengde p� edge i 3D
    CXx = (x2-x1)/L; % lengde p� edge i x retning / total lengde
    CYx = (y2-y1)/L; % lengde p� edge i y retning / total lengde
    CZx = (z2-z1)/L; % lengde p� edge i z retning / total lengde
    T = EModulus*A(e)/L*[CXx*CXx CXx*CYx CXx*CZx ; CYx*CXx CYx*CYx CYx*CZx ; CZx*CXx CZx*CYx CZx*CZx]; % [3,3] matrise
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DIAGONAL 1
    IJindexTeller = IJindexTeller+1;
    % celle x1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(1); % x index
    J( IJindexTeller ) = edgeDof(1); % y index
    X( IJindexTeller ) = T(1,1); % verdi
    
    IJindexTeller = IJindexTeller+1;
    % celle y1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(1); % x index
    J( IJindexTeller ) = edgeDof(2); % y index
    X( IJindexTeller ) = T(1,2); % verdi
    
    IJindexTeller = IJindexTeller+1;
    % celle z1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(1); % x index
    J( IJindexTeller ) = edgeDof(3); % y index
    X( IJindexTeller ) = T(1,3); % verdi
    
    
    IJindexTeller = IJindexTeller+1;
    % celle x1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(2); % x index
    J( IJindexTeller ) = edgeDof(1); % y index
    X( IJindexTeller ) = T(2,1); % verdi
    
    IJindexTeller = IJindexTeller+1;
    % celle y1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(2); % x index
    J( IJindexTeller ) = edgeDof(2); % y index
    X( IJindexTeller ) = T(2,2); % verdi
    
    IJindexTeller = IJindexTeller+1;
    % celle z1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(2); % x index
    J( IJindexTeller ) = edgeDof(3); % y index
    X( IJindexTeller ) = T(2,3); % verdi
    
    
    IJindexTeller = IJindexTeller+1;
    % celle x1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(3); % x index
    J( IJindexTeller ) = edgeDof(1); % y index
    X( IJindexTeller ) = T(3,1); % verdi
    
    IJindexTeller = IJindexTeller+1;
    % celle y1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(3); % x index
    J( IJindexTeller ) = edgeDof(2); % y index
    X( IJindexTeller ) = T(3,2); % verdi
    
    IJindexTeller = IJindexTeller+1;
    % celle z1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(3); % x index
    J( IJindexTeller ) = edgeDof(3); % y index
    X( IJindexTeller ) = T(3,3); % verdi
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DIAGONAL 2
    IJindexTeller = IJindexTeller+1;
    % celle x1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(4); % x index
    J( IJindexTeller ) = edgeDof(4); % y index
    X( IJindexTeller ) = T(1,1); % verdi
    
    IJindexTeller = IJindexTeller+1;
    % celle y1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(4); % x index
    J( IJindexTeller ) = edgeDof(5); % y index
    X( IJindexTeller ) = T(1,2); % verdi
    
    IJindexTeller = IJindexTeller+1;
    % celle z1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(4); % x index
    J( IJindexTeller ) = edgeDof(6); % y index
    X( IJindexTeller ) = T(1,3); % verdi
    
    
    IJindexTeller = IJindexTeller+1;
    % celle x1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(5); % x index
    J( IJindexTeller ) = edgeDof(4); % y index
    X( IJindexTeller ) = T(2,1); % verdi
    
    IJindexTeller = IJindexTeller+1;
    % celle y1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(5); % x index
    J( IJindexTeller ) = edgeDof(5); % y index
    X( IJindexTeller ) = T(2,2); % verdi
    
    IJindexTeller = IJindexTeller+1;
    % celle z1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(5); % x index
    J( IJindexTeller ) = edgeDof(6); % y index
    X( IJindexTeller ) = T(2,3); % verdi
    
    
    IJindexTeller = IJindexTeller+1;
    % celle x1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(6); % x index
    J( IJindexTeller ) = edgeDof(4); % y index
    X( IJindexTeller ) = T(3,1); % verdi
    
    IJindexTeller = IJindexTeller+1;
    % celle y1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(6); % x index
    J( IJindexTeller ) = edgeDof(5); % y index
    X( IJindexTeller ) = T(3,2); % verdi
    
    IJindexTeller = IJindexTeller+1;
    % celle z1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(6); % x index
    J( IJindexTeller ) = edgeDof(6); % y index
    X( IJindexTeller ) = T(3,3); % verdi
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% oppe h�yre bolk
    IJindexTeller = IJindexTeller+1;
    % celle x1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(1); % x index
    J( IJindexTeller ) = edgeDof(4); % y index
    X( IJindexTeller ) = -T(1,1); % verdi
    
    IJindexTeller = IJindexTeller+1;
    % celle y1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(1); % x index
    J( IJindexTeller ) = edgeDof(5); % y index
    X( IJindexTeller ) = -T(1,2); % verdi
    
    IJindexTeller = IJindexTeller+1;
    % celle z1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(1); % x index
    J( IJindexTeller ) = edgeDof(6); % y index
    X( IJindexTeller ) = -T(1,3); % verdi
    
    
    IJindexTeller = IJindexTeller+1;
    % celle x1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(2); % x index
    J( IJindexTeller ) = edgeDof(4); % y index
    X( IJindexTeller ) = -T(2,1); % verdi
    
    IJindexTeller = IJindexTeller+1;
    % celle y1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(2); % x index
    J( IJindexTeller ) = edgeDof(5); % y index
    X( IJindexTeller ) = -T(2,2); % verdi
    
    IJindexTeller = IJindexTeller+1;
    % celle z1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(2); % x index
    J( IJindexTeller ) = edgeDof(6); % y index
    X( IJindexTeller ) = -T(2,3); % verdi
    
    
    IJindexTeller = IJindexTeller+1;
    % celle x1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(3); % x index
    J( IJindexTeller ) = edgeDof(4); % y index
    X( IJindexTeller ) = -T(3,1); % verdi
    
    IJindexTeller = IJindexTeller+1;
    % celle y1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(3); % x index
    J( IJindexTeller ) = edgeDof(5); % y index
    X( IJindexTeller ) = -T(3,2); % verdi
    
    IJindexTeller = IJindexTeller+1;
    % celle z1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(3); % x index
    J( IJindexTeller ) = edgeDof(6); % y index
    X( IJindexTeller ) = -T(3,3); % verdi
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% nedre venstre bolk
    IJindexTeller = IJindexTeller+1;
    % celle x1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(4); % x index
    J( IJindexTeller ) = edgeDof(1); % y index
    X( IJindexTeller ) = -T(1,1); % verdi
    
    IJindexTeller = IJindexTeller+1;
    % celle y1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(4); % x index
    J( IJindexTeller ) = edgeDof(2); % y index
    X( IJindexTeller ) = -T(1,2); % verdi
    
    IJindexTeller = IJindexTeller+1;
    % celle z1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(4); % x index
    J( IJindexTeller ) = edgeDof(3); % y index
    X( IJindexTeller ) = -T(1,3); % verdi
    
    
    IJindexTeller = IJindexTeller+1;
    % celle x1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(5); % x index
    J( IJindexTeller ) = edgeDof(1); % y index
    X( IJindexTeller ) = -T(2,1); % verdi
    
    IJindexTeller = IJindexTeller+1;
    % celle y1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(5); % x index
    J( IJindexTeller ) = edgeDof(2); % y index
    X( IJindexTeller ) = -T(2,2); % verdi
    
    IJindexTeller = IJindexTeller+1;
    % celle z1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(5); % x index
    J( IJindexTeller ) = edgeDof(3); % y index
    X( IJindexTeller ) = -T(2,3); % verdi
    
    
    IJindexTeller = IJindexTeller+1;
    % celle x1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(6); % x index
    J( IJindexTeller ) = edgeDof(1); % y index
    X( IJindexTeller ) = -T(3,1); % verdi
    
    IJindexTeller = IJindexTeller+1;
    % celle y1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(6); % x index
    J( IJindexTeller ) = edgeDof(2); % y index
    X( IJindexTeller ) = -T(3,2); % verdi
    
    IJindexTeller = IJindexTeller+1;
    % celle z1 bort, x1 ned
    I( IJindexTeller ) = edgeDof(6); % x index
    J( IJindexTeller ) = edgeDof(3); % y index
    X( IJindexTeller ) = -T(3,3); % verdi
    
    % kl�sjer en 6x6 matrise ofte opp� i pos bestemt av edge noder
    %K(edgeDof,edgeDof) = K(edgeDof,edgeDof) + EModulus*A(e)/L*[T -T ; -T T];
end
if verbose
    fprintf('tripl2sparse ... ');
end

KK = sparse(I,J,X,kDof,kDof);

% TEST
% T = CXx*CXx CXx*CYx CXx*CZx
%     CYx*CXx CYx*CYx CYx*CZx
%     CZx*CXx CZx*CYx CZx*CZx
% clear;
% K=zeros(9,9);
% T=[1 2 3 ; 4 5 6 ; 7 8 9]
%
%      1     2     3
%      4     5     6
%      7     8     9
% [T -T ; -T T]
%      1     2     3    -1    -2    -3
%      4     5     6    -4    -5    -6
%      7     8     9    -7    -8    -9
%     -1    -2    -3     1     2     3
%     -4    -5    -6     4     5     6
%     -7    -8    -9     7     8     9

% K([1 2 3 6 7 8],[1 2 3 6 7 8]) = [T -T ; -T T]

%      1     2     3     0     0    -1    -2    -3     0
%      4     5     6     0     0    -4    -5    -6     0
%      7     8     9     0     0    -7    -8    -9     0
%      0     0     0     0     0     0     0     0     0
%      0     0     0     0     0     0     0     0     0
%     -1    -2    -3     0     0     1     2     3     0
%     -4    -5    -6     0     0     4     5     6     0
%     -7    -8    -9     0     0     7     8     9     0
%      0     0     0     0     0     0     0     0     0
%
% Full sparse
% extF =
%
%      1     2     3
%      4     5     6
%      7     8     9
%
%    I, J        X
%    (1,1)        1
%    (2,1)        4
%    (3,1)        7
%    (1,2)        2
%    (2,2)        5
%    (3,2)        8
%    (1,3)        3
%    (2,3)        6
%    (3,3)        9
%
% https://books.google.no/books?id=oovDyJrnr6UC&pg=PA170&lpg=PA170&dq=making+fem+sparse+matrix+from+triplets+matlab&source=bl&ots=rQcFUyE-GI&sig=ACfU3U2fmxfhihhixEdEUUQk1Fnjh-jTzQ&hl=en&sa=X&ved=2ahUKEwiimIvX3u3nAhXd4KYKHbaDA9g4ChDoATAAegQIChAB#v=onepage&q=making%20fem%20sparse%20matrix%20from%20triplets%20matlab&f=false
% http://milamin.org/technical-notes/sparse-matrix-assembly/
% https://books.google.no/books?id=pCa-DwAAQBAJ&pg=PA586&lpg=PA586&dq=making+fem+sparse+stiffness+matrix+from+triplets+matlab&source=bl&ots=imuKQSw860&sig=ACfU3U22mu-G_kD5nV0Lre60_2SMgcQ1Ag&hl=en&sa=X&ved=2ahUKEwif6b7x3-3nAhWCyMQBHfl5AuIQ6AEwCnoECAgQAQ#v=onepage&q=making%20fem%20sparse%20stiffness%20matrix%20from%20triplets%20matlab&f=false

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% L�SER

% fjerner rader og kolonner som forsvinner grunnet boundary
% https://stackoverflow.com/questions/4163876/removing-rows-and-columns-from-matlab-matrix-quickly

c = find(extC==0);

if verbose
    fprintf('fjerner rad/col med boundary ... ');
end
%KK = KK(bb,bb);

KK(c,:)=[];
KK(:,c)=[];
extF(c,:)=[];

% KKK = zeros(size(KK) - [numel(b) numel(b)]);
% r = true(size(KK,1),1);
% r(b) = false;
% KKK = KK(r,r);

if verbose
    tic;
end

if verbose
    fprintf('CPU start ... ');
end

%full(KK)
U = KK \ extF ; % nodal displ = stiffness / force
%NrNaN = sum(isnan(U(:)));
%while NrNaN > 0
%    KK = KK + rand(size(extF)).*1e-20;
%    U = KK \ extF;
%    NrNaN = sum(isnan(U(:)));
%end
    
spparms('spumoni',0); % eller 2


if verbose
    fprintf('CPU ferdig \n');
end

if verbose
    toc;
end

% setter tilbake rader og kolonner som er fjernet grunnet boundary
cc = find(extC==1); % setter bare inn p� indekser der det er noe, ellers 0
Ufull=zeros(numberOfNodes*3,1);
for i=1:length(cc)
    Ufull(cc(i),1)= U(i,1);
end

dU = reshape(Ufull, 3, [])';

%%%%%%%%%%%%%%%%%%%%%%%%%%%% STRESS

eS=zeros(numberOfEdges,1);
for e=1:numberOfEdges
    indice=E(e,:);
    edgeDof = [3*indice(1)-2 3*indice(1)-1 3*indice(1)   3*indice(2)-2 3*indice(2)-1 3*indice(2)]; %index til current Edge
    x1=N(indice(1),1);
    y1=N(indice(1),2);
    z1=N(indice(1),3);
    x2=N(indice(2),1);
    y2=N(indice(2),2);
    z2=N(indice(2),3);
    L = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1)); % opprinnelige L
    CXx = (x2-x1)/L;
    CYx = (y2-y1)/L;
    CZx = (z2-z1)/L;
    uc = Ufull(edgeDof); % u til current Edge
    eS(e) = EModulus/L * [-CXx -CYx -CZx CXx CYx CZx]*uc;
end

% ikke 100% lik
% eS2=zeros(numberOfEdges,1);
% for e=1:numberOfEdges
%     indice=E(e,:);
%     x1 = N(indice(1),1);
%     y1 = N(indice(1),2);
%     z1 = N(indice(1),3);
%     x2 = N(indice(2),1);
%     y2 = N(indice(2),2);
%     z2 = N(indice(2),3);
%     L = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1)); % opprinnelige L
%
%     x1ny = N(indice(1),1) + U(indice(1),1);
%     y1ny = N(indice(1),2) + U(indice(1),2);
%     z1ny = N(indice(1),3) + U(indice(1),3);
%     x2ny = N(indice(2),1) + U(indice(2),1);
%     y2ny = N(indice(2),2) + U(indice(2),2);
%     z2ny = N(indice(2),3) + U(indice(2),3);
%     Lny = sqrt((x2ny-x1ny)*(x2ny-x1ny) + (y2ny-y1ny)*(y2ny-y1ny) + (z2ny-z1ny)*(z2ny-z1ny)); % opprinnelige L
%
%     eS2(e) = EModulus * (Lny - L)/L;
% end

%%
    function readVgrarginInput()
        while ~isempty(varargin)
            switch varargin{1}
                case 'eMod'
                    EModulus = varargin{2}; varargin(1:2) = [];
                case 'verb'
                    verbose = 1;varargin(1:1) = [];
                otherwise
                    error(['Unexpected option: ' varargin{1}]);
            end
        end
    end

end