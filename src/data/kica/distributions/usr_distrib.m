function outp=usr_distrib(dist_name,query,param)
% USR_DISTRIB - user defined distributions, as used in the Kernel ICA paper
%
% dist_name     - a letter between 'a' and 'r'
% query,param   - either 'rnd',  param is then the number of samples
%                        'pdf',  param is then a row of abcissas
%                        'name', param is then optional
%

% Copyright (c) Francis R. Bach, 2002.

switch dist_name
   
case 'm' % mixture of 4 Gaussians, symmetric and multimodal
   prop=[1 2 2 1];
   prop=prop/sum(prop);
   mus=[-1 -.33 .33 1];
   covs=[.16  .16 .16 .16];
   switch query
   case 'name'
      outp='Mix4Gauss_SymMultiModal';
   case 'pdf'
      outp=0;
      for i=1:4
         outp=outp+prop(i)*my_normpdf(param,mus(i),covs(i));
      end
   case 'rnd'
      for j=1:param
         i=sample_discrete(prop, 1, 1);
         outp(j)=mynormrnd(mus(i),covs(i));
      end
   case 'kurt'
      mu=0;
      x2=0;
      x4=0;
      x3=0;
      for i=1:length(prop)
         mu=mu+prop(i)*mus(i);
         x2=x2+prop(i)*(mus(i)^2+covs(i)^2);
         x3=x3+prop(i)*(3*mus(i)*covs(i)^2+mus(i)^3);
         x4=x4+prop(i)*(3*covs(i)^4+6*covs(i)^2*mus(i)^2+mus(i)^4);
      end
      outp=(x4-4*mu*x3+6*mu^2*x2-3*mu^4)/(x2-mu^2)^2-3;
      
   end
   
case 'n' % mixture of 4 Gaussians, symmetric and transitional
   prop=[1 2 2 1];
   prop=prop/sum(prop);
   mus=[-1 -.2 .2 1];
   covs=[.2  .3 .3 .2];
   switch query
   case 'name'
      outp='Mix4Gauss_SymTransitional';
   case 'pdf'
      outp=0;
      for i=1:4
         outp=outp+prop(i)*my_normpdf(param,mus(i),covs(i));
      end
   case 'rnd'
      for j=1:param
         i=sample_discrete(prop, 1, 1);
         outp(j)=mynormrnd(mus(i),covs(i));
      end
   case 'kurt'
      mu=0;
      x2=0;
      x4=0;
      x3=0;
      for i=1:length(prop)
         mu=mu+prop(i)*mus(i);
         x2=x2+prop(i)*(mus(i)^2+covs(i)^2);
         x3=x3+prop(i)*(3*mus(i)*covs(i)^2+mus(i)^3);
         x4=x4+prop(i)*(3*covs(i)^4+6*covs(i)^2*mus(i)^2+mus(i)^4);
      end
      outp=(x4-4*mu*x3+6*mu^2*x2-3*mu^4)/(x2-mu^2)^2-3;
      
      
   end
   
   
   
   
case 'o' % mixture of 4 Gaussians, symmetric and unimodal
   prop=[1 2 2 1];
   prop=prop/sum(prop);
   mus=[-.7 -.2 .2 .7];
   covs=[.2  .3 .3 .2];
   switch query
   case 'name'
      outp='Mix4Gauss_SymUniModal';
   case 'pdf'
      outp=0;
      for i=1:4
         outp=outp+prop(i)*my_normpdf(param,mus(i),covs(i));
      end
   case 'rnd'
      for j=1:param
         i=sample_discrete(prop, 1, 1);
         outp(j)=mynormrnd(mus(i),covs(i));
      end
   case 'kurt'
      mu=0;
      x2=0;
      x4=0;
      x3=0;
      for i=1:length(prop)
         mu=mu+prop(i)*mus(i);
         x2=x2+prop(i)*(mus(i)^2+covs(i)^2);
         x3=x3+prop(i)*(3*mus(i)*covs(i)^2+mus(i)^3);
         x4=x4+prop(i)*(3*covs(i)^4+6*covs(i)^2*mus(i)^2+mus(i)^4);
      end
      outp=(x4-4*mu*x3+6*mu^2*x2-3*mu^4)/(x2-mu^2)^2-3;
      
      
   end
   
   
   
case 'p' % mixture of 4 Gaussians, nonsymmetric and multimodal
   prop=[1 1 2 1];
   prop=prop/sum(prop);
   mus=[-1 .3 -.3 1.1];
   covs=[.2  .2 .2 .2];
   switch query
   case 'name'
      outp='Mix4Gauss_AssymMultiModal';
   case 'pdf'
      outp=0;
      for i=1:4
         outp=outp+prop(i)*my_normpdf(param,mus(i),covs(i));
      end
   case 'rnd'
      for j=1:param
         i=sample_discrete(prop, 1, 1);
         outp(j)=mynormrnd(mus(i),covs(i));
      end
   case 'kurt'
      mu=0;
      x2=0;
      x4=0;
      x3=0;
      for i=1:length(prop)
         mu=mu+prop(i)*mus(i);
         x2=x2+prop(i)*(mus(i)^2+covs(i)^2);
         x3=x3+prop(i)*(3*mus(i)*covs(i)^2+mus(i)^3);
         x4=x4+prop(i)*(3*covs(i)^4+6*covs(i)^2*mus(i)^2+mus(i)^4);
      end
      outp=(x4-4*mu*x3+6*mu^2*x2-3*mu^4)/(x2-mu^2)^2-3;
      
      
   end
   
   
   
   
case 'q' % mixture of 4 Gaussians, nonsymmetric and transitional
   prop=[1 3 2 .5];
   prop=prop/sum(prop);
   mus=[-1 -.2 .3 1];
   covs=[.2  .3 .2 .2];
   switch query
   case 'name'
      outp='Mix4Gauss_AssymTransitional';
   case 'pdf'
      outp=0;
      for i=1:4
         outp=outp+prop(i)*my_normpdf(param,mus(i),covs(i));
      end
   case 'rnd'
      for j=1:param
         i=sample_discrete(prop, 1, 1);
         outp(j)=mynormrnd(mus(i),covs(i));
      end
   case 'kurt'
      mu=0;
      x2=0;
      x4=0;
      x3=0;
      for i=1:length(prop)
         mu=mu+prop(i)*mus(i);
         x2=x2+prop(i)*(mus(i)^2+covs(i)^2);
         x3=x3+prop(i)*(3*mus(i)*covs(i)^2+mus(i)^3);
         x4=x4+prop(i)*(3*covs(i)^4+6*covs(i)^2*mus(i)^2+mus(i)^4);
      end
      outp=(x4-4*mu*x3+6*mu^2*x2-3*mu^4)/(x2-mu^2)^2-3;
      
      
   end
   
   
   
case 'r' % mixture of 4 Gaussians, nonsymmetric and unimodal
   prop=[1 2 2 1];
   prop=prop/sum(prop);
   mus=[-.8 -.2 .2 .5];
   covs=[.22  .3 .3 .2];
   switch query
   case 'name'
      outp='Mix4Gauss_AssymUniModal';
   case 'pdf'
      outp=0;
      for i=1:4
         outp=outp+prop(i)*my_normpdf(param,mus(i),covs(i));
      end
   case 'rnd'
      for j=1:param
         i=sample_discrete(prop, 1, 1);
         outp(j)=mynormrnd(mus(i),covs(i));
      end
   case 'kurt'
      mu=0;
      x2=0;
      x4=0;
      x3=0;
      for i=1:length(prop)
         mu=mu+prop(i)*mus(i);
         x2=x2+prop(i)*(mus(i)^2+covs(i)^2);
         x3=x3+prop(i)*(3*mus(i)*covs(i)^2+mus(i)^3);
         x4=x4+prop(i)*(3*covs(i)^4+6*covs(i)^2*mus(i)^2+mus(i)^4);
      end
      outp=(x4-4*mu*x3+6*mu^2*x2-3*mu^4)/(x2-mu^2)^2-3;
      
      
   end
   
   
   
   
   
   
   
   
   
   
   
case 'a' % Student T with 3 degrees of freedom
   switch query
   case 'name'
      outp='Student_3deg';
   case 'pdf'
      outp=tpdf(param,3);
   case 'rnd'
      outp=mytrnd(3,1,param);
   case 'kurt'
      outp=Inf;
   end
   
   
   
case 'b' % double exponential
   switch query
   case 'name'
      outp='DbleExponential';
   case 'pdf'
      outp=exp(-sqrt(2)*abs(param))/sqrt(2);
   case 'rnd'
      outp=sign(rand(1,param)-.5).*myexprnd(1/sqrt(2),1,param);
   case 'kurt'
      outp=3;
   end
   
   
   
case 'c' % Uniform
   switch query
   case 'name'
      outp='Uniform';
   case 'pdf'
      outp=1/2/sqrt(3).*(param<sqrt(3)).*(param>-sqrt(3));
   case 'rnd'
      outp=-sqrt(3)+rand(1,param)*2*sqrt(3);
   case 'kurt'
      outp=-1.2;
   end
   
   
   
   
   
case 'g' % mixture of 2 Gaussians, symmetric and multimodal
   prop=[.5 .5];
   mus=[-.5 .5 ];
   covs=[.15  .15];
   switch query
   case 'name'
      outp='Mix2Gauss_SymMultiModal';
   case 'pdf'
      outp=0;
      for i=1:2
         outp=outp+prop(i)*my_normpdf(param,mus(i),covs(i));
      end
   case 'rnd'
      for j=1:param
         i=sample_discrete(prop, 1, 1);
         outp(j)=mynormrnd(mus(i),covs(i));
      end
   case 'kurt'
      mu=0;
      x2=0;
      x4=0;
      x3=0;
      for i=1:length(prop)
         mu=mu+prop(i)*mus(i);
         x2=x2+prop(i)*(mus(i)^2+covs(i)^2);
         x3=x3+prop(i)*(3*mus(i)*covs(i)^2+mus(i)^3);
         x4=x4+prop(i)*(3*covs(i)^4+6*covs(i)^2*mus(i)^2+mus(i)^4);
      end
      outp=(x4-4*mu*x3+6*mu^2*x2-3*mu^4)/(x2-mu^2)^2-3;
      
      
   end
   
   
   
   
case 'h' % mixture of 2 Gaussians, symmetric and transitional
   prop=[.5 .5];
   mus=[-.5 .5];
   covs=[.4  .4];
   switch query
   case 'name'
      outp='Mix2Gauss_SymTransitional';
   case 'pdf'
      outp=0;
      for i=1:2
         outp=outp+prop(i)*my_normpdf(param,mus(i),covs(i));
      end
   case 'rnd'
      for j=1:param
         i=sample_discrete(prop, 1, 1);
         outp(j)=mynormrnd(mus(i),covs(i));
      end
   case 'kurt'
      mu=0;
      x2=0;
      x4=0;
      x3=0;
      for i=1:length(prop)
         mu=mu+prop(i)*mus(i);
         x2=x2+prop(i)*(mus(i)^2+covs(i)^2);
         x3=x3+prop(i)*(3*mus(i)*covs(i)^2+mus(i)^3);
         x4=x4+prop(i)*(3*covs(i)^4+6*covs(i)^2*mus(i)^2+mus(i)^4);
      end
      outp=(x4-4*mu*x3+6*mu^2*x2-3*mu^4)/(x2-mu^2)^2-3;
      
      
   end
   
   
   
   
case 'i' % mixture of 2 Gaussians, symmetric and unimodal
   prop=[.5 .5];
   mus=[-.5 .5];
   covs=[.5  .5];
   switch query
   case 'name'
      outp='Mix2Gauss_SymUniModal';
   case 'pdf'
      outp=0;
      for i=1:2
         outp=outp+prop(i)*my_normpdf(param,mus(i),covs(i));
      end
   case 'rnd'
      for j=1:param
         i=sample_discrete(prop, 1, 1);
         outp(j)=mynormrnd(mus(i),covs(i));
      end
   case 'kurt'
      mu=0;
      x2=0;
      x4=0;
      x3=0;
      for i=1:length(prop)
         mu=mu+prop(i)*mus(i);
         x2=x2+prop(i)*(mus(i)^2+covs(i)^2);
         x3=x3+prop(i)*(3*mus(i)*covs(i)^2+mus(i)^3);
         x4=x4+prop(i)*(3*covs(i)^4+6*covs(i)^2*mus(i)^2+mus(i)^4);
      end
      outp=(x4-4*mu*x3+6*mu^2*x2-3*mu^4)/(x2-mu^2)^2-3;
      
      
   end
   
   
   
   
   
   
   
   
   
   
case 'j' % mixture of 2 Gaussians, nonsymmetric and multimodal
   prop=[1 3];
   prop=prop/sum(prop);
   mus=[-.5 .5 ];
   covs=[.15  .15];
   switch query
   case 'name'
      outp='Mix2Gauss_AssymMultiModal';
   case 'pdf'
      outp=0;
      for i=1:2
         outp=outp+prop(i)*prop(i)*my_normpdf(param,mus(i),covs(i));
      end
   case 'rnd'
      for j=1:param
         i=sample_discrete(prop, 1, 1);
         outp(j)=mynormrnd(mus(i),covs(i));
      end
   case 'kurt'
      mu=0;
      x2=0;
      x4=0;
      x3=0;
      for i=1:length(prop)
         mu=mu+prop(i)*mus(i);
         x2=x2+prop(i)*(mus(i)^2+covs(i)^2);
         x3=x3+prop(i)*(3*mus(i)*covs(i)^2+mus(i)^3);
         x4=x4+prop(i)*(3*covs(i)^4+6*covs(i)^2*mus(i)^2+mus(i)^4);
      end
      outp=(x4-4*mu*x3+6*mu^2*x2-3*mu^4)/(x2-mu^2)^2-3;
      
      
   end
   
   
   
   
case 'k' % mixture of 2 Gaussians, nonsymmetric and transitional
   prop=[1 2];
   prop=prop/sum(prop);
   mus=[-.7 .5];
   covs=[.4  .4];
   switch query
   case 'name'
      outp='Mix2Gauss_AssymTransitional';
   case 'pdf'
      outp=0;
      for i=1:2
         outp=outp+prop(i)*my_normpdf(param,mus(i),covs(i));
      end
   case 'rnd'
      for j=1:param
         i=sample_discrete(prop, 1, 1);
         outp(j)=mynormrnd(mus(i),covs(i));
      end
   case 'kurt'
      mu=0;
      x2=0;
      x4=0;
      x3=0;
      for i=1:length(prop)
         mu=mu+prop(i)*mus(i);
         x2=x2+prop(i)*(mus(i)^2+covs(i)^2);
         x3=x3+prop(i)*(3*mus(i)*covs(i)^2+mus(i)^3);
         x4=x4+prop(i)*(3*covs(i)^4+6*covs(i)^2*mus(i)^2+mus(i)^4);
      end
      outp=(x4-4*mu*x3+6*mu^2*x2-3*mu^4)/(x2-mu^2)^2-3;
      
      
   end
   
   
   
   
case 'l' % mixture of 2 Gaussians, nonsymmetric and unimodal
   prop=[1 2];
   prop=prop/sum(prop);
   mus=[-.7 .5 ];
   covs=[.5  .5];
   switch query
   case 'name'
      outp='Mix2Gauss_AssymUniModal';
   case 'pdf'
      outp=0;
      for i=1:2
         outp=outp+prop(i)*my_normpdf(param,mus(i),covs(i));
      end
   case 'rnd'
      for j=1:param
         i=sample_discrete(prop, 1, 1);
         outp(j)=mynormrnd(mus(i),covs(i));
      end
   case 'kurt'
      mu=0;
      x2=0;
      x4=0;
      x3=0;
      for i=1:length(prop)
         mu=mu+prop(i)*mus(i);
         x2=x2+prop(i)*(mus(i)^2+covs(i)^2);
         x3=x3+prop(i)*(3*mus(i)*covs(i)^2+mus(i)^3);
         x4=x4+prop(i)*(3*covs(i)^4+6*covs(i)^2*mus(i)^2+mus(i)^4);
      end
      outp=(x4-4*mu*x3+6*mu^2*x2-3*mu^4)/(x2-mu^2)^2-3;
      
      
   end
   
   
   
   
   
   
   
   
   
case 'd' % Student T with 5 degrees of freedom
   switch query
   case 'name'
      outp='Student_5deg';
   case 'pdf'
      outp=tpdf(param,5);
   case 'rnd'
      outp=mytrnd(5,1,param);
   case 'kurt'
      outp=6;
   end
   
   
case 'e' % simple exponential
   switch query
   case 'name'
      outp='Exponential';
   case 'pdf'
      outp=exp(-(param+1)).*(param>-1);
   case 'rnd'
      outp=-1+myexprnd(1,1,param);
   case 'kurt'
      outp=6;
   end
   
   
case 'f'  % mixtures of 2 double exponential
   spdf.funct='pdfmixdble_exp';
   spdf.rnd='rndmixdble_exp';
   prop=[.5 .5];
   mus=[-1 1];
   covs=[ .5 .5];
   switch query
   case 'name'
      outp='Mix2DbleExp';
   case 'pdf'
      outp=0;
      for i=1:2
         outp=outp+prop(i)/covs(i)*exp(-sqrt(2)*abs(param-mus(i))/covs(i))/sqrt(2);
      end
   case 'rnd'
      for j=1:param
         i=sample_discrete(prop, 1, 1);
         outp(j)=sign(rand-.5).*myexprnd(1/sqrt(2))*covs(i)+mus(i);
      end
      
   case 'kurt'
      for i=1:length(mus)
         mus(i)=mus(i)*covs(i);
      end
      
      mu=0;
      x2=0;
      x4=0;
      x3=0;
      for i=1:length(prop)
         mu=mu+prop(i)*mus(i);
         x2=x2+prop(i)*(mus(i)^2+covs(i));
         x3=x3+prop(i)*(3*mus(i)*covs(i)+mus(i)^3);
         x4=x4+prop(i)*(6*covs(i)^2+6*covs(i)*mus(i)^2+mus(i)^4);
      end
      outp=(x4-4*mu*x3+6*mu^2*x2-3*mu^4)/(x2-mu^2)^2-3;
      
      
   end
   
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function M = sample_discrete(prob, r, c)
% SAMPLE_DISCRETE Like the built in 'rand', except we draw from a non-uniform discrete distrib.
% M = sample_discrete(prob, r, c)
%

% Written by Kevin Murphy,  murphyk@cs.berkeley.edu
% Copyright (c) University of California 1997-2001 

if nargin == 1
   r = 1; c = 1;
elseif nargin == 2
   c == r;
end

% this speedup is due to Peter Acklam
cumprob = cumsum(prob(:));
n = length(cumprob);
R = rand(r, c);
M = ones(r, c);
for i = 1:n-1
   M = M + (R > cumprob(i));
end


