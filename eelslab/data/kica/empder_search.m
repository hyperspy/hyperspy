function [Jopt,Wopt,OptDetails] = empder_search(contrast,x,W,kparam,optparam)
% EMPDER_SEARCH - Steepest descent method for finding a minima in the
%                 Stiefel manifold of orthogonal matrices, using empirical
%                 derivatives. Data are assumed whitened. The output is such 
%                 that Wopt*x are the independent sources.
%
%
% contrast   - used contrast function, 'kcca' or 'kgv'
% x          - data (mixtures)
% w          - orthogonal matric, starting point of the search
% kparam     - contrast parameters, see contrast_ica.m for details
% optparam   - optimization parameters
%                  tolW    : precision in amari distance in input space
%                  tolJ    : precision in objective function
%                  maxit   : maximum number of iterations
%                  type    : 'steepest' or 'conjugate'
%                  verbose : 1 if verbose required.
%
% OptDetails - optional output, with debugging details

% Copyright (c) Francis R. Bach, 2002.

% initializations
W=W'; % we work with transposed demixing matrices
if (nargout>2), details=1; else details=0; end
tolW=optparam.tolW;
tolJ=optparam.tolJ;
maxit=optparam.maxit;
type=optparam.type;
verbose=optparam.verbose;
tmin=1;
iter = 0;
errW = tolW*2;
errJ = tolJ*2;
m=size(W,1);
fret = contrast_ica(contrast,W'*x,kparam);
totalneval=1;
transgradJ=0;

% starting minimization
while (((errW > tolW)|(errJ > tolJ*fret)) & (iter < maxit)  )
   Jold=fret;
   iter=iter+1;
   if (verbose), fprintf('iter %d, J=%.3f',iter,fret); end
   
   % calculate derivative
   [J0,gradJ]=contrast_emp_grad(contrast,x,kparam,W);
   iterneval=m*(m-1)/2+1;
   normgradJ=sqrt(.5*trace(gradJ'*gradJ));
   
   dirSearch=gradJ; 
   normdirSearch=sqrt(.5*trace(gradJ'*gradJ));
   
   % bracketing the minimum along the geodesic and performs golden search
   [ ax, bx, cx,fax,fbx,fcx,neval] = bracket_min(contrast,W,dirSearch,x,kparam,0,tmin,Jold);
   iterneval=iterneval+neval;
   goldTol=max(abs([tolW/normdirSearch, mean([ ax, bx, cx])/10]));
   [tmin, Jmin,neval] = golden_search(contrast,W,dirSearch,x,kparam, ax, bx, cx,goldTol,20);
   iterneval=iterneval+neval;
   if (verbose)
      fprintf(', dJ= %.1e',Jold-Jmin);
      fprintf(', dW= %.3f, neval=%d\n',tmin*normdirSearch,iterneval);
   end
   totalneval=totalneval+iterneval;
   oldtransgradJ=transgradJ;
   Wnew=stiefel_geod(W,dirSearch,tmin);  
   oldnormgradJ=sqrt(.5*trace(gradJ'*gradJ));
   
   errW=amari_distance(W,Wnew)*(m-1);
   
   if (details)
      % debugging details
      OptDetails.Ws{iter}=W;
      OptDetails.Js(iter)=J0;
      OptDetails.numeval(iter)=totalneval;
      OptDetails.numgoldit(iter)=neval;
      OptDetails.ts(iter)=tmin;
      OptDetails.normdirsearch(iter)=normdirSearch;
      OptDetails.normgrad(iter)=oldnormgradJ;
      OptDetails.amaridist(iter)=errW;
      OptDetails.geoddist(iter)=tmin*normdirSearch;
   end
   
   errJ=Jold-Jmin;
   if (errJ>0) 
      W=Wnew;
      fret=Jmin;
   end
   
end

Jopt= fret;
Wopt=W'; % go back to non transposed matrices

if (details)
   OptDetails.totalneval=totalneval;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [Wt,Ht]=stiefel_geod(W,H,t)

% STIEFEL_GEOD - parameterizes a geodesic along a Stiefel manifold

% W  - origin of the geodesic
% H  - tangent vector
% Ht - tangent vector at "arrival"

% Copyright (c) Francis R. Bach, 2002.


if nargin <3, t=1; end
A=W'*H; A=(A-A')/2;
MN=expm(t*A);
Wt=W*MN;
if nargout > 1, Ht=H*MN; end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [xmin,fmin,neval] = golden_search(contrast,W,dirT,x,kparam,ax,bx,cx,tol,maxiter)

% GOLDEN_SEARCH - Minimize contrast function along a geodesic of the Stiefel
%                 manifold using golden section search.
%
% contrast       - contrast function used, 'kcca' or 'kgv'
% x              - mixed components
% kparam         - contrast parameters, with following fields
%                        sigmas - kernel widths (one per component)
%                        kappas - regularization parameters (one per component)
%                        etas   - incomplete Cholesky tolerance (one per component)
%
% dirT           - direction of the geodesic
% ax,bx,cx       - three abcissas such that the minimum is bracketed between ax and cx,
%                  as given by bracket_mini.m
% tol            - relative accuracy of the search
% maxit          - maximum number of iterations

% neval          - outputs the number of evaluation of the contrast function


neval=0;
% golden ratios
C = (3-sqrt(5))/2;
R = 1-C;

x0 = ax;
x3 = cx;

% gets the smaller segment
if (abs(cx-bx) > abs(bx-ax)),
   x1 = bx;
   x2 = bx + C*(cx-bx);
else
   x2 = bx;
   x1 = bx - C*(bx-ax);
end
Wtemp=stiefel_geod(W,dirT,x1);
f1=contrast_ica(contrast,Wtemp'*x,kparam);
neval=neval+1;
Wtemp=stiefel_geod(W,dirT,x2);
f2=contrast_ica(contrast,Wtemp'*x,kparam);
neval=neval+1;
k = 1;

% starts iterations
while ((abs(x3-x0) > tol) & (k<maxiter)), 
   if f2 < f1,
      x0 = x1;
      x1 = x2;
      x2 = R*x1 + C*x3;   
      f1 = f2;
      Wtemp=stiefel_geod(W,dirT,x2);
      f2=contrast_ica(contrast,Wtemp'*x,kparam);
      neval=neval+1;
   else
      x3 = x2;
      x2 = x1;
      x1 = R*x2 + C*x0;  
      f2 = f1;
      Wtemp=stiefel_geod(W,dirT,x1);
      f1=contrast_ica(contrast,Wtemp'*x,kparam);
      neval=neval+1;
   end
   k = k+1;
end

% best of the two possible
if f1 < f2,
   xmin = x1;
   fmin = f1;
else
   xmin = x2;
   fmin = f2;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ ax, bx, cx,fax,fbx,fcx,neval] = bracket_min(contrast,W,dirT,x,kparam, ax, bx,fax)

% BRACKET_MIN - Brackets a minimum by searching in both directions along a geodesic in
%               the Stiefel manifold

% contrast       - contrast function used, 'kcca' or 'kgv'
% x              - mixed components
% kparam         - contrast parameters, with following fields
%                        sigmas - kernel widths (one per component)
%                        kappas - regularization parameters (one per component)
%                        etas   - incomplete Cholesky tolerance (one per component)
%
% dirT           - direction of the geodesic
% ax,bx          - Initial guesses
% tol            - relative accuracy of the search
% maxit          - maximum number of iterations

% neval          - outputs the number of evaluation of the contrast function


neval=0;
GOLD=1.618034;
TINY=1e-10;
GLIMIT=100;
Wtemp=stiefel_geod(W,dirT,bx);
fbx=contrast_ica(contrast,Wtemp'*x,kparam);

neval=neval+1;

if (fbx > fax)   
   temp=ax;
   ax=bx;
   bx=temp;
   temp=fax;
   fax=fbx;
   fbx=temp;
end

cx=(bx)+GOLD*(bx-ax);
Wtemp=stiefel_geod(W,dirT,cx);
fcx=contrast_ica(contrast,Wtemp'*x,kparam);

neval=neval+1;

while (fbx > fcx) 
   
   r=(bx-ax)*(fbx-fcx);
   q=(bx-cx)*(fbx-fax);
   u=(bx)-((bx-cx)*q-(bx-ax)*r)/(2.0*max([abs(q-r),TINY])*sign(q-r));
   ulim=(bx)+GLIMIT*(cx-bx);
   if ((bx-u)*(u-cx) > 0.0)
      Wtemp=stiefel_geod(W,dirT,u);
      fux=contrast_ica(contrast,Wtemp'*x,kparam);
      
      neval=neval+1;
      
      if (fux < fcx) 
         ax=(bx);
         bx=u;
         fax=(fbx);
         fbx=fux;
         return;
      else 
         if (fux > fbx) 
            cx=u;
            fcx=fux;
            return;
         end
      end
      
      u=(cx)+GOLD*(cx-bx);
      Wtemp=stiefel_geod(W,dirT,u);
      fux=contrast_ica(contrast,Wtemp'*x,kparam);
      neval=neval+1;
      
   else 
      if ((cx-u)*(u-ulim) > 0.0) 
         Wtemp=stiefel_geod(W,dirT,u);
         fux=contrast_ica(contrast,Wtemp'*x,kparam);
         neval=neval+1;
         
         if (fux < fcx) 
            bx=cx;
            cx=u;
            u=cx+GOLD*(cx-bx);
            
            fbx=fcx;
            fcx=fux;
            Wtemp=stiefel_geod(W,dirT,u);
            fux=contrast_ica(contrast,Wtemp'*x,kparam);
         end
      else 
         if ((u-ulim)*(ulim-cx) >= 0.0) 
            
            u=ulim;
            Wtemp=stiefel_geod(W,dirT,u);
            fux=contrast_ica(contrast,Wtemp'*x,kparam);
            neval=neval+1;
            
         else 
            u=(cx)+GOLD*(cx-bx);
            Wtemp=stiefel_geod(W,dirT,u);
            fux=contrast_ica(contrast,Wtemp'*x,kparam);
            neval=neval+1;
            
         end
      end
   end
   
   ax=bx;
   bx=cx;
   cx=u;
   
   fax=fbx;
   fbx=fcx;
   fcx=fux;
   
end


