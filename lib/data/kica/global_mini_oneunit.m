function [Jopt,Wopt,OptDetails] = global_mini_oneunit(contrast,x,W,kparam,optparam);
% GLOBAL_MINI_ONEUNIT
%            - global minimization of contrast function with random restarts
%              the data are assumed whitened (i.e. with identity covariance
%              matrix). The output is such that Wopt*x are the independent
%              sources.
%
% contrast   - used contrast function, 'kcca' or 'kgv'
% x          - data (mixtures)
% w          - orthogonal matric, starting point of the search
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


%initialization
tolW=optparam.tolW;
tolJ=optparam.tolJ;
maxit=optparam.maxit;
type=optparam.type;
Nrestart=optparam.Nrestarts;
verbose=optparam.verbose;
Jaccept=optparam.Jaccept;
if (nargout>2), details=1; else details=0; end
iter = 1;
m=size(W,1);
Wmin=W;
Jmin= contrast_ica_oneunit(contrast,x,W,kparam);
totalneval=1;

%starting restarts loop
while (iter<=Nrestart) & (Jmin>Jaccept)
   if (verbose) fprintf('\nStarting a new local search, #%d\n',iter); end
   
   if (iter>1) 
      % selects a new random restart as far as possible as current minimums
      NWs=m*m*4;
      Wrandrest=cell(1,NWs);
      distances=zeros(1,2*iter-2);
      maxmindist=0;
      W=[];
      for i=1:NWs
         Wrandrest=rand(m,1)-.5;
         Wrandrest=Wrandrest/norm(Wrandrest);
         for j=1:iter-1
            distances(2*j-1)=norm(Wrandrest-OptDetails.Wloc{j});
            distances(2*j  )=norm(Wrandrest-OptDetails.Wstart{j});
         end
         mindist=min(distances);
         if (mindist>maxmindist) maxmindist=mindist; W=Wrandrest; end
      end
   end
   
   % performs local search to local minimum (requires a non transposed matrix)
   newkparam=kparam;
    switch(newkparam.kernel)
   case 'hermite'
      newkparam.ps=newkparam.ps-sign(newkparam.ps-2);
   case 'poly'
      newkparam.ds=newkparam.ds-sign(newkparam.ps-2);
   case 'gaussian'
      newkparam.sigmas=newkparam.sigmas*2;
   end
   [Jloc,W,detailsloc] = empder_search_oneunit(contrast,x,W,newkparam,optparam);
   [Jloc,Wloc,detailsloc] = empder_search_oneunit(contrast,x,W,kparam,optparam);
   if (iter==1)
      Wmin=Wloc;
      Jmin=Jloc;
   else
      if (Jloc<Jmin), Wmin=Wloc; Jmin=Jloc; end
   end
   
   totalneval=totalneval+detailsloc.totalneval;
   OptDetails.Wloc{iter}=Wloc;
   OptDetails.Wstart{iter}=W;
   OptDetails.Jloc(iter)=Jloc;
   iter=iter+1;
end
Jopt= Jmin;
Wopt=Wmin;