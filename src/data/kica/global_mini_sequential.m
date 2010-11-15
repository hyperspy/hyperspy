function [Jopts,Wopt,OptDetails] = global_mini_sequential(contrast,x,W,kparam,optparam);
% GLOBAL_MINI_SEQUENTIAL
%             - global minimization of contrast function with random restarts
%               the data are assumed whitened (i.e. with identity covariance
%               matrix). The output is such that Wopt*x are the independent
%               sources. It uses one-unit contrast functions in a deflation
%               scheme. See paper for details.
%
% contrast   - used contrast function, 'kcca' or 'kgv'
% x          - data (mixtures)
% W          - orthogonal matric, starting point of the search
% kparam     - contrast parameters, see contrast_ica.m for details
% optparam   - optimization parameters
%                  tolW      : precision in amari distance in input space
%                  tolJ      : precision in objective function
%                  maxit     : maximum number of iterations
%                  type      : 'steepest' or 'conjugate'
%                  verbose   : 1 if verbose required.
%                  Nrestarts : number of restarts
%
% OptDetails - optional output, with debugging details

% Copyright (c) Francis R. Bach, 2002.



if (nargout>2), details=1; else details=0; end
verbose=optparam.verbose;

m=size(W,1);
Wc=W';
xc=Wc'*x;
ws=cell(1,m-1);
worths=cell(1,m-1);
Jopts=zeros(1,m-1);
idim=m;
jdim=1;
while (idim>1);
   if (verbose) 
      fprintf('\nStarting search for a univariate component in %d dim\n',idim);
   end
   winit=rand(idim,1)-1;
   winit=winit/norm(winit);
   [newJ,wnew,OptDetails] = global_mini_oneunit(contrast,xc,winit,kparam,optparam);
   [a,b]=sort(OptDetails.Jloc);
   WS=[];
   for j=1:length(OptDetails.Wloc)
      WS=[WS OptDetails.Wloc{j}];
   end
   
   
   % determines which components will be kept
   tokeep=b(1);
   for j=2:length(b)
      % want an IC that is orthogonal to all precious IC
      prodscals=WS(:,b(j))'*WS(:,tokeep);
      if (norm(prodscals,inf)<0.1), tokeep=[tokeep b(j)]; end 
   end
   
   Jopts(jdim)=newJ;
   ws{jdim}=WS(:,tokeep);
   wnewc=null(ws{jdim}');
   worths{jdim}=wnewc;
   xc=wnewc'*xc;
   idim=size(wnewc,2);
   jdim=jdim+1;
end
jdim=jdim-1;

%reconstruct
Wopt=Wc';
for i=1:jdim;
   idim=size(ws{i},1);
   Wopt=[eye(m-idim) zeros(m-idim,idim); zeros(idim,m-idim) [ws{i} worths{i}]']*Wopt;
end

