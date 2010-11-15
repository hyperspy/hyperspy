function [J0,gradJ]=contrast_emp_grad_oneunit(contrast,x,kparam,w)
% CONTRAST_EMP_GRAD_ONE_UNIT - Evaluation of the derivatives of the one-unit
%                              contrast function using empirical derivatives
%

% Copyright (c) Francis R. Bach, 2002.

w0=w;
m=length(w);
[J0,details]=contrast_ica_oneunit(contrast,x,w,kparam);
wc=details.wc;
WTgradF=zeros(m,1);
dr=0.001;
for j=2:m
   wdr=w0*cos(dr)+wc(:,j-1)*sin(dr);
   J=contrast_update_oneunit(contrast,x,w0,kparam,j,dr,details);
   WTgradF=WTgradF+(J-J0)/dr*wc(:,j-1);
end
gradJ=WTgradF;


