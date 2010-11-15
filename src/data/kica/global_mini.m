function [Jopt,Wopt,OptDetails] = global_mini(contrast,x,W,kparam,optparam);
% GLOBAL_MINI - global minimization of contrast function with random restarts
%               the data are assumed whitened (i.e. with identity covariance
%               matrix). The output is such that Wopt*x are the independent
%               sources.
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
W=W'; % we work with transposed demixing matrices
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
Jmin= contrast_ica(contrast,W'*x,kparam);
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
         Wrandrest=rand_orth(m);
         for j=1:iter-1
            distances(2*j-1)=amari_distance(Wrandrest,OptDetails.Wloc{j});
            distances(2*j  )=amari_distance(Wrandrest,OptDetails.Wstart{j});
         end
         mindist=min(distances);
         if (mindist>maxmindist) maxmindist=mindist; W=Wrandrest; end
      end
   end
   
   % performs local search to local minimum (requires a non transposed matrix)
   [Jloc,Wloc,detailsloc] = empder_search(contrast,x,W',kparam,optparam);
   Wloc=Wloc'; % goes back to transposed matrix
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
Wopt=Wmin'; % we go back to non transposed demixing matrices


