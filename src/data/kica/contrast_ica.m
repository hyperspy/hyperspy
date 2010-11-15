function [J,details]=contrast_ica(contrast,x,kparam);
% CONTRAST_ICA   - compute the Kernel-ICA contrast function based on
%                    kernel canonical correlation analysis
%
% contrast       - contrast function used, 'kcca' or 'kgv'
% x              - mixed components
% kparam         - contrast parameters, with following fields
%                        kappas - regularization parameters (one per component)
%                        etas   - incomplete Cholesky tolerance (one per component)
%                        kernel - type of kernel: 'gaussian', 'poly', 'spline'
%
%                        sigmas - kernel widths (one per component) for translation
%                                 invariant kernels
%                        rs,ss,ds - polynomial kernel parameters (r+s*x'*y)^d
%
% details        - optional output with details of the decomposition
%                - as used by update_contrast.m

% Copyright (c) Francis R. Bach, 2002.






N=size(x,2);		% number of data points
m=size(x,1);      % number of components
kappas=kparam.kappas;
etas=kparam.etas;
Rkappa=[];
sizes=[];
for i=1:m
   % cholesky decomposition using a MEX-file
   switch (kparam.kernel)
   case 'hermite'
     [G,Pvec] =chol_hermite(x(i,:),kparam.sigmas(i),kparam.ps(i),N*etas(i)); 
   case 'dirichlet'
      [G,Pvec] =chol_dirichlet(x(i,:),kparam.sigmas(i),kparam.ps(i),N*etas(i)); 
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
case 'kcca'
   OPTIONS.disp=0;
   OPTIONS.tol=1e-5;
   if (nargout>1)
      [beta,D]=eigs(Rkappa,1,'SM',OPTIONS);
      J=-.5*log(D);
      
      % outputs details
      details.Us=Us;
      details.Lambdas=Lambdas;
      details.Rkappa=Rkappa;
      details.beta=beta;
      details.Drs=Drs;
      details.sizes=sizes;
      details.starts=starts;
      j1=1;
      for i=1:m
         betas{i}=beta(j1:j1+size(Us{i},2)-1);
         j1=j1+size(Us{i},2);
      end
      details.betas=betas;
   else
      D=eigs(Rkappa,1,'SM',OPTIONS);
      J=-.5*log(D);
   end
   
case 'kgv'
   D=det(Rkappa);     
   J=-.5*log(D);
   if (nargout>1)
      % outputs details
      details.Us=Us;
      details.Lambdas=Lambdas;
      details.Drs=Drs;
      details.Rkappa=Rkappa;
      details.sizes=sizes;
      details.starts=starts;
   end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function G2=centerpartial(G1)
% CENTERPARTIAL - Center a gram matrix of the form K=G*G'

[N,NG]=size(G1);
G2 = G1 - repmat(mean(G1,1),N,1);



