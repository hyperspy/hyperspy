function [J,details]=contrast_upgrade_oneunit(contrast,x,w,kparam,jjj,dr,details);
% CONTRAT_UPGRADE_ONEUNIT - compute the Kernel-ICA contrast function based on
%                           kernel canonical correlation analysis, for one unit
%                           contrast functions
%
% contrast       - contrast function used, 'kcca', 'kgv'
% x              - mixed components
% kparam         - contrast parameters, with following fields
%                        kappas - regularization parameters (one per component)
%                        etas   - incomplete Cholesky tolerance (one per component)
%                        kernel - type of kernel: 'gaussian', 'poly', 'spline'

%                        sigmas - kernel widths (one per component) for translation
%                                 invariant kernels
%                        rs,ss,ds - polynomial kernel parameters (r+s*x'*y)^d

% details        - optional output with details of the decomposition
%                - as used by update_contrast.m

% Copyright (c) Francis R. Bach, 2002.

N=size(x,2);		% number of data points
m=size(x,1);      % number of components



wc=details.wc;
% only needs to update the first one and the jjj-th one
neww0=w*cos(dr)+wc(:,jjj-1)*sin(dr);
newwj=wc(:,jjj-1)*cos(dr)-w*sin(dr);

kappas=kparam.kappas;
etas=kparam.etas;
Rkappa=details.Rkappa;
Us=details.Us;
Lambdas=details.Lambdas;
Drs=details.Drs;
sizes=details.sizes;
oldstarts=details.starts;
oldsizes=sizes;

% redo the two cholesky decompositions using a MEX-file
for i=[1 jjj]
   if (i==1)
      tochol=neww0'*x;
   else
      tochol=newwj'*x;
   end
   
   switch (kparam.kernel)
   case 'hermite'
      [G,Pvec] =chol_hermite(tochol,kparam.sigmas(i),kparam.ps(i),N*etas(i)); 
   case 'gaussian'
      [G,Pvec] =chol_gauss(tochol/kparam.sigmas(i),1,N*etas(i)); 
   case 'poly'
      [G,Pvec] =chol_poly(tochol,kparam.rs(i),kparam.ss(i),kparam.ds(i),N*etas(i)); 
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
newRkappa=eye(sum(sizes));

for i=2:m
   for j=1:i-1
      if ( (j==1) | (i==jjj) | (j==jjj) )
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
case 'kgv'
   J=-.5*log(det(newRkappa));
   J=J+.5*log(det(newRkappa(starts(2):starts(m)+sizes(m)-1,starts(2):starts(m)+sizes(m)-1)));
case 'kcca'
   M22=chol(newRkappa(starts(2):starts(m)+sizes(m)-1,starts(2):starts(m)+sizes(m)-1));
   invM22=inv(M22);
   prepostmult=[eye(sizes(1)) zeros(sizes(1),length(newRkappa)-sizes(1)); ...
         zeros(length(newRkappa)-sizes(1),sizes(1)) invM22];
   OPTIONS.disp=0;
   OPTIONS.tol=1e-5;
   D=eigs(prepostmult'*newRkappa*prepostmult,1,'SM',OPTIONS);
   J=-.5*log(D);   
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function G2=centerpartial(G1)
% CENTERPARTIAL - Center a gram matrix of the form K=G*G'

[N,NG]=size(G1);
G2 = G1 - repmat(mean(G1,1),N,1);



