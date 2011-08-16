function J=update_contrast(contrast,x,kparam,details,ii,jj);
% UPDATE_CONTRAST      - compute the Kernel-ICA contrast function based on
%                        kernel canonical correlation analysis starting from
%                        already calculated details, useful for calculating 
%                        empirical gradient
%
% contrast       - contrast function used, 'kcca' or 'kgv'
% x              - mixed components
% kparam         - contrast parameters, with following fields
%                        sigmas - kernel widths (one per component)
%                        kappas - regularization parameters (one per component)
%                        etas   - incomplete Cholesky tolerance (one per component)

% Copyright (c) Francis R. Bach, 2002.




N=size(x,2);		% number of data points
m=size(x,1);      % number of components

% ensures that ii is less than jj
if (ii>jj), te=ii;  ii=jj;  jj=te; end

kappas=kparam.kappas;
etas=kparam.etas;

% download details
Rkappa=details.Rkappa;
Us=details.Us;
Lambdas=details.Lambdas;
Drs=details.Drs;
sizes=details.sizes;
oldstarts=details.starts;
oldsizes=sizes;
%Updates Us, Lambdas, Drs for indices ii and jj
for i=[ii jj]
   % cholesky decomposition using a MEX-file
   switch (kparam.kernel)
        case 'spline'
        case 'hermite'
     [G,Pvec] =chol_hermite(x(i,:),kparam.sigmas(i),kparam.ps(i),N*etas(i)); 

   case 'dirichlet'
      [G,Pvec] =chol_dirichlet(x(i,:),kparam.sigmas(i),kparam.ps(i),1,N*etas(i)); 
   case 'exponential'
      [G,Pvec] =chol_expo(x(i,:)/kparam.sigmas(i),1,N*etas(i)); 
   case 'gaussian'
      [G,Pvec] =chol_gauss(x(i,:)/kparam.sigmas(i),1,N*etas(i)); 
   case 'poly'
      [G,Pvec] =chol_poly(x(i,:),kparam.rs(i),kparam.ss(i),kparam.ds(i),N*etas(i)); 
   end
   [a,Pvec]=sort(Pvec);
   G=centerpartial(G(Pvec,:));
   
   % regularization (see paper for details)
   [A,D]=eig(G'*G);
   D=diag(D);
   indexes=find(D>=N*etas(i) & isreal(D)); %removes small eigenvalues
   [newinds,order]=sort(D(indexes));
   order=flipud(order);
   neig=length(indexes);
   indexes=indexes(order(1:neig));  
   if (isempty(indexes)), indexes=[1]; end
   D=D(indexes);
   V=G*(A(:,indexes)*diag(sqrt(1./(D))));
   Us{i}=V;
   Lambdas{i}=D;
   Dr=D;
   for j=1:length(D)
      Dr(j)=D(j)/(N*kappas(i)+D(j));
   end
   Drs{i}=Dr;
   sizes(i)=size(Drs{i},1);
end

starts=cumsum([1 sizes]);
starts(m+1)=[];


% now creates a new Rkappa, we know that ii is less than jj
newRkappa=eye(sum(sizes));
for i=2:m
   for j=1:i-1
      if ( (j==ii) | (i==jj) | (i==ii) | (j==jj) )
         newbottom=diag(Drs{i})*(Us{i}'*Us{j})*diag(Drs{j});
         newRkappa(starts(i):starts(i)+sizes(i)-1,starts(j):starts(j)+sizes(j)-1)=newbottom;
         newRkappa(starts(j):starts(j)+sizes(j)-1,starts(i):starts(i)+sizes(i)-1)=newbottom';
      else
         newbottom= Rkappa(oldstarts(i):oldstarts(i)+oldsizes(i)-1,oldstarts(j):oldstarts(j)+oldsizes(j)-1);
         newRkappa(starts(i):starts(i)+sizes(i)-1,starts(j):starts(j)+sizes(j)-1)=newbottom;
         newRkappa(starts(j):starts(j)+sizes(j)-1,starts(i):starts(i)+sizes(i)-1)=newbottom';
      end
   end
end

switch contrast
case 'kcca'
   OPTIONS.disp=0;
   OPTIONS.tol=1e-5;
   D=eigs(newRkappa,1,'SM',OPTIONS);
   J=-.5*log(D);
case 'kgv'
   D=det(newRkappa);     
   J=-.5*log(D);
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function G2=centerpartial(G1)
% CENTERPARTIAL - Center a gram matrix of the form K=G*G'

[N,NG]=size(G1);
G2 = G1 - repmat(mean(G1,1),N,1);
