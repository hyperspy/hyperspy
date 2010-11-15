function [J,details]=contrast_ica_oneunit(contrast,x,w,kparam,details);

% CONTRAT_ICA_ONEUNIT - compute the Kernel-ICA contrast function based on
%                       kernel canonical correlation analysis, for one unit
%                       contrast functions
%
% contrast       - contrast function used, 'kcca', 'kgv'
% x              - mixed components
% kparam         - contrast parameters, with following fields
%                        kappas - regularization parameters (one per component)
%                        etas   - incomplete Cholesky tolerance (one per component)
%                        kernel - type of kernel: 'gaussian', 'poly', 'hermite'
%
%                        sigmas - kernel widths (one per component) for translation
%                                 invariant kernels
%                        rs,ss,ds - polynomial kernel parameters (r+s*x'*y)^d
%                        sigmas,ps - hermite kernel parameter.
%
% details        - optional output with details of the decomposition
%                - as used by update_contrast.m

% Copyright (c) Francis R. Bach, 2002.

N=size(x,2);		% number of data points
m=size(x,1);      % number of components

% first compute a specific orthogonal complement
w=w/norm(w);
e=zeros(m,1);
e(1)=1;
id=eye(m);
if abs(1-e'*w)<1e-12
   wc=id(:,2:m);
else
   p=e'*w;
   q=sqrt(1-p^2);
   a=(w*-p+e)/q;
   wac=null([w a]');
   Pb=[w a wac];
   R=[p q; -q p];
   rotmat=Pb*[ R zeros(2,m-2) ; zeros(m-2,2) eye(m-2)]*Pb';
   wc=rotmat*id(:,2:m);
end



W=[w wc];
s=W'*x;
kappas=kparam.kappas;
etas=kparam.etas;
Rkappa=[];
sizes=[];
for i=1:m
   % cholesky decomposition using a MEX-file
   switch (kparam.kernel)
   case 'hermite'
      [G,Pvec] =chol_hermite(s(i,:),kparam.sigmas(i),kparam.ps(i),N*etas(i)); 
   case 'gaussian'
      [G,Pvec] =chol_gauss(s(i,:)/kparam.sigmas(i),1,N*etas(i)); 
   case 'poly'
      [G,Pvec] =chol_poly(s(i,:),kparam.rs(i),kparam.ss(i),kparam.ds(i),N*etas(i)); 
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

% calculated Rkappa
Rkappa=eye(sum(sizes));
starts=cumsum([1 sizes]);
starts(m+1)=[];
for i=2:m
   for j=1:(i-1)
      newbottom=diag(Drs{i})*(Us{i}'*Us{j})*diag(Drs{j});
      Rkappa(starts(i):starts(i)+sizes(i)-1,starts(j):starts(j)+sizes(j)-1)=newbottom;
      Rkappa(starts(j):starts(j)+sizes(j)-1,starts(i):starts(i)+sizes(i)-1)=newbottom';
   end
end

switch contrast
case 'kgv'
   J=-.5*log(det(Rkappa));
   J=J+.5*log(det(Rkappa(starts(2):starts(m)+sizes(m)-1,starts(2):starts(m)+sizes(m)-1)));
   if (nargout>1)
      % outputs details
      details.Us=Us;
      details.Lambdas=Lambdas;
      details.Drs=Drs;
      details.Rkappa=Rkappa;
      details.sizes=sizes;
      details.starts=starts;
   end
   
   
case 'kcca'
   M22=chol(Rkappa(starts(2):starts(m)+sizes(m)-1,starts(2):starts(m)+sizes(m)-1));
   invM22=inv(M22);
   prepostmult=[eye(sizes(1)) zeros(sizes(1),length(Rkappa)-sizes(1)); ...
         zeros(length(Rkappa)-sizes(1),sizes(1)) invM22];
   OPTIONS.disp=0;
   OPTIONS.tol=1e-5;
   [beta,D]=eigs(prepostmult'*Rkappa*prepostmult,1,'SM',OPTIONS);
   J=-.5*log(D);
   if (nargout>1)
      details.Us=Us;
      details.Lambdas=Lambdas;
      details.Rkappa=Rkappa;
      details.beta=beta;
      details.Drs=Drs;
      details.sizes=sizes;
      details.starts=starts;
      
   end
   
   
end



if (nargout>1)
   details.wc=wc;
   details.rotmat=rotmat;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function G2=centerpartial(G1)
% CENTERPARTIAL - Center a gram matrix of the form K=G*G'

[N,NG]=size(G1);
G2 = G1 - repmat(mean(G1,1),N,1);



