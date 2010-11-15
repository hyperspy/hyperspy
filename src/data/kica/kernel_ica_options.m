function W=kernel_ica_options(x,varargin)

% KERNEL_ICA - Runs the kernel ica algorithm on the given mixtures.
%              Returns a demixing matrix W such that s=W*x are the 
%              independent components
%
%	            It first centers and whitens the data and then minimizes
%              the given contrast function over orthogonal matrices.
%
%
%
% OPTIONAL PARAMETERS
%
%    Field name        Parameter                             Default
%
%    'contrast'        contrast function                     'kgv'
%                      values: 'kcca' 'kgv'
%
%    'contrasttype'    contrast type, values: 					 'full'
%                      'full' 'oneunit'
%
%    'ncomp'           Number of desired univariate          m
%                      components, 1 or m          
%
%    'polish'          1 if finishing with a half sigma      1
%                      value (better estimate)
%
%    'restarts'        number of restarts                    1
%
%    'kernel'          kernel type, 'gaussian', 'poly'       'gaussian'
%                      'hermite'
%    'sig'             bandwidth for gaussian kernel         0.5
%    'r','s','d'       parameters for polynomial kernel      1,1,3
%                      
%    'p','sig'         parameters for Dirichlet, Hermite     4,2
%
%    'kap'             regularization parameter              0.01
%    'W0'              demixing matric initialization        rand_orth(m)
%    'disp'            verbose intermediate results          1







% default values

optimization='steepest';
contrast='kcca';
contrasttype='full';
verbose=1;
[m,N]=size(x);
ncomp=m;

% currently fixed values but should depend on N and m
sigma=1;
rpol=1;
spol=1;
dpol=3;
pherm=4;
kappa=.01;
eta=0.0001;
Nrestarts=1;
polish=1;
kernel='gaussian';
W0=rand_orth(m);

if (rem(length(varargin),2)==1)
   error('Optional parameters should always go by pairs');
else
   for i=1:2:(length(varargin)-1)
      switch varargin{i}
      case 'contrast'
         contrast= varargin{i+1};
      case 'contrasttype'
         contrasttype= varargin{i+1};
      case 'sig'
         sigma=varargin{i+1};
      case 'r'
         rpol=varargin{i+1};
      case 'ncomp'
         ncomp=varargin{i+1};
      case 's'
         spol=varargin{i+1};
      case 'd'
         dpol=varargin{i+1};
      case 'kap'
         kappa=varargin{i+1};
      case 'p'
         pherm=varargin{i+1};
      case 'kernel'
         kernel=varargin{i+1};
      case 'W0'
         W0=varargin{i+1};
      case 'disp'
         verbose=varargin{i+1};
      case 'polish'
         polish=varargin{i+1};
      case 'restarts'
         Nrestarts=varargin{i+1};
         
      end
   end
end

% definition of parameters
mc=m;
kparam.kappas=kappa*ones(1,mc);
kparam.etas=kappa*1e-2*ones(1,mc);
kparam.neigs=N*ones(1,mc);
kparam.nchols=N*ones(1,mc);
kparam.kernel=kernel;
switch(kernel)
case 'hermite'
   kparam.sigmas=sigma*ones(1,mc);
   kparam.ps=pherm*ones(1,mc);
case 'poly'
   kparam.rs=rpol*ones(1,mc);
   kparam.ss=spol*ones(1,mc);
   kparam.ds=dpol*ones(1,mc);
case 'gaussian'
   kparam.sigmas=sigma*ones(1,mc);
end


% first centers and scales data
if (verbose), fprintf('centering and scaling...'); end
xc=x-repmat(mean(x,2),1,N);  % centers data
covmat=xc*xc'/N;
sqcovmat=sqrtm(covmat);
invsqcovmat=inv(sqcovmat);
xc=invsqcovmat*xc;           % scales data
if (verbose), fprintf('done\n'); end


optparam.tolW=1e-2;
optparam.tolJ=1e-2;
optparam.maxit=20;
optparam.type=optimization;
optparam.verbose=verbose;
if (ncomp==m)
   % making initial guess orthogonal (for a full matrix)
   [U,S,V]=svd(W0*sqcovmat);
   W0=U*V';
else
   % or unit norm (for one component)
   if (size(W0,2)>1), W0=rand(m,1)-.5; end
   W0=sqcovmat*W0;
   W0=W0/norm(W0);
end

if (verbose), fprintf('Starting optimization, with %d restarts\n',Nrestarts); end
optparam.Nrestarts=Nrestarts;
optparam.Jaccept=0;



if isequal(contrasttype,'full')
   [J,W,details]= global_mini(contrast,xc,W0,kparam,optparam);
else
   if (ncomp==1)
      [J,W,details]= global_mini_oneunit(contrast,xc,W0,kparam,optparam);
   else
      [J,W,details]= global_mini_sequential(contrast,xc,W0,kparam,optparam);
   end
   
end




% polishing: 1. using finer contrast functions, i.e:
%               	-with smaller bandwidth (Gaussian kernels)
%               	-higher order (polynomial, Hermite kernels)
%            2. using full contrast function.


if (polish)
   if (verbose), fprintf('\nPolishing...\n'); end
   switch(kernel)
   case 'hermite'
      kparam.ps= kparam.ps+1;
   case 'poly'
      kparam.ds=kparam.ds+1;
   case 'gaussian'
      kparam.sigmas=kparam.sigmas/2;
   end
   [J,W,details] = empder_search(contrast,xc,W,kparam,optparam);
end


if (ncomp==m)
   W=W*invsqcovmat;
else
   W=invsqcovmat*W;
end

