function [Jopt,wopt,OptDetails] = empder_search_oneunit(contrast,x,w,kparam,optparam)

% EMPDER_SEARCH_ONEUNIT
%               - Steepest descent method for finding a minima of the one-unit
%                 contrast function in the manifold of unit vectors, using empirical
%                 derivatives. Data are assumed whitened. The output is such 
%                 that w'*x is one of the independent sources.
%
% contrast   - used contrast function, 'kcca' or 'kgv'
% x          - data (mixtures)
% w          - unit vector, starting point of the search
% kparam     - contrast parameters, see contrast_ica_oneunit.m for details
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
m=length(w);
fret = contrast_ica_oneunit(contrast,x,w,kparam);
totalneval=1;
transgradJ=0;

% starting minimization
while (((errW > tolW)|(errJ > tolJ*fret)) & (iter < maxit)  )
   Jold=fret;
   iter=iter+1;
   if (verbose), fprintf('iter %d, J=%.3f',iter,fret); end
   
   % calculate derivative (a vector in our case)
   [J0,gradJ]=contrast_emp_grad_oneunit(contrast,x,kparam,w);
   iterneval=m+1;
   normgradJ=norm(gradJ'*gradJ);
   dirSearch=gradJ; 
   normdirSearch=sqrt(gradJ'*gradJ);
   
   % bracketing the minimum along the geodesic and performs golden search
   [ ax, bx, cx,fax,fbx,fcx,neval] = bracket_min(contrast,w,dirSearch,x,kparam,0,tmin,Jold);
   iterneval=iterneval+neval;
   goldTol=max(abs([tolW/normdirSearch, mean([ ax, bx, cx])/10]));
   [tmin, Jmin,neval] = golden_search(contrast,w,dirSearch,x,kparam, ax, bx, cx,goldTol,20);
   iterneval=iterneval+neval;
   if (verbose)
      fprintf(', dJ= %.1e',Jold-Jmin);
      fprintf(', dW= %.3f, neval=%d\n',tmin*normdirSearch,iterneval);
   end
   totalneval=totalneval+iterneval;
   oldtransgradJ=transgradJ;
   
   wnew=w*cos(norm(dirSearch)*tmin)+dirSearch/norm(dirSearch)*sin(norm(dirSearch)*tmin);
   
   oldnormgradJ=sqrt(gradJ'*gradJ);
   
   errW=abs(norm(dirSearch)*tmin);
   
   
   if (details)
      % debugging details
      OptDetails.ws{iter}=w;
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
      w=wnew;
      fret=Jmin;
   end
   
end

Jopt= fret;
wopt=w;
if (details)
   OptDetails.totalneval=totalneval;
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [xmin,fmin,neval] = golden_search(contrast,w,dirT,x,kparam,ax,bx,cx,tol,maxiter)

% GOLDEN_SEARCH - Minimize contrast function along a geodesic of the unit 
%                 sphere using golden section search.
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
wtemp=w*cos(norm(dirT)*x1)+dirT/norm(dirT)*sin(norm(dirT)*x1);
f1=contrast_ica_oneunit(contrast,x,wtemp,kparam);
neval=neval+1;

wtemp=w*cos(norm(dirT)*x2)+dirT/norm(dirT)*sin(norm(dirT)*x2);
f2=contrast_ica_oneunit(contrast,x,wtemp,kparam);

neval=neval+1;
k = 1;

% starts iterations
while ((abs(x3-x0) > tol) & (k<maxiter)), 
   if f2 < f1,
      x0 = x1;
      x1 = x2;
      x2 = R*x1 + C*x3;   
      f1 = f2;
      wtemp=w*cos(norm(dirT)*x2)+dirT/norm(dirT)*sin(norm(dirT)*x2);
      f2=contrast_ica_oneunit(contrast,x,wtemp,kparam);
      neval=neval+1;
   else
      x3 = x2;
      x2 = x1;
      x1 = R*x2 + C*x0;  
      f2 = f1;
      wtemp=w*cos(norm(dirT)*x1)+dirT/norm(dirT)*sin(norm(dirT)*x1);
      f1=contrast_ica_oneunit(contrast,x,wtemp,kparam);
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

function [ ax, bx, cx,fax,fbx,fcx,neval] = bracket_min(contrast,w,dirT,x,kparam, ax, bx,fax)

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
wtemp=w*cos(norm(dirT)*bx)+dirT/norm(dirT)*sin(norm(dirT)*bx);
fbx=contrast_ica_oneunit(contrast,x,wtemp,kparam);

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
wtemp=w*cos(norm(dirT)*cx)+dirT/norm(dirT)*sin(norm(dirT)*cx);
fcx=contrast_ica_oneunit(contrast,x,wtemp,kparam);

neval=neval+1;

while (fbx > fcx) 
   
   r=(bx-ax)*(fbx-fcx);
   q=(bx-cx)*(fbx-fax);
   u=(bx)-((bx-cx)*q-(bx-ax)*r)/(2.0*max([abs(q-r),TINY])*sign(q-r));
   ulim=(bx)+GLIMIT*(cx-bx);
   if ((bx-u)*(u-cx) > 0.0)
      wtemp=w*cos(norm(dirT)*u)+dirT/norm(dirT)*sin(norm(dirT)*u);
      fux=contrast_ica_oneunit(contrast,x,wtemp,kparam);
      
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
      wtemp=w*cos(norm(dirT)*u)+dirT/norm(dirT)*sin(norm(dirT)*u);
      fux=contrast_ica_oneunit(contrast,x,wtemp,kparam);
      neval=neval+1;
      
   else 
      if ((cx-u)*(u-ulim) > 0.0) 
         wtemp=w*cos(norm(dirT)*u)+dirT/norm(dirT)*sin(norm(dirT)*u);
         fux=contrast_ica_oneunit(contrast,x,wtemp,kparam);
         neval=neval+1;
         
         if (fux < fcx) 
            bx=cx;
            cx=u;
            u=cx+GOLD*(cx-bx);
            
            fbx=fcx;
            fcx=fux;
            wtemp=w*cos(norm(dirT)*u)+dirT/norm(dirT)*sin(norm(dirT)*u);
            fux=contrast_ica_oneunit(contrast,x,wtemp,kparam);
         end
      else 
         if ((u-ulim)*(ulim-cx) >= 0.0) 
            
            u=ulim;
            wtemp=w*cos(norm(dirT)*u)+dirT/norm(dirT)*sin(norm(dirT)*u);
            fux=contrast_ica_oneunit(contrast,x,wtemp,kparam);
            neval=neval+1;
            
         else 
            u=(cx)+GOLD*(cx-bx);
            wtemp=w*cos(norm(dirT)*u)+dirT/norm(dirT)*sin(norm(dirT)*u);
            fux=contrast_ica_oneunit(contrast,x,wtemp,kparam);
            
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


