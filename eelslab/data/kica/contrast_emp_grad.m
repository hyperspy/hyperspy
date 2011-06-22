function [J0,gradJ]=contrast_emp_grad(contrast,x,kparam,W)
% CONTRAST_EMP_GRAD - Evaluation of the derivatives of the m-way criterion,
%                     using empirical derivatives
%
% same input as contrast_ica.m

% Copyright (c) Francis R. Bach, 2002.

W0=W;
m=size(W);
WTgradF=zeros(m);
s0=W'*x;
[J0,details]=contrast_ica(contrast, s0,kparam);
dr=0.001;
for i=1:m-1
   for j=i+1:m
      s=s0;
      s([i j],:)=[cos(dr) sin(dr) ; sin(-dr) cos(dr)]*s([i j],:);
      J=update_contrast(contrast,s,kparam,details,i,j);
      WTgradF(i,j)=(J-J0)/dr;
      WTgradF(j,i)=-(J-J0)/dr;
   end
end

gradJ=W0*WTgradF;


  
