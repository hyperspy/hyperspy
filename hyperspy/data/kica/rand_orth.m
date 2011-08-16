function W=rand_orth(n,m);
% RAND_ORTH - random matrix with orthogonal columns

% Copyright (c) Francis R. Bach, 2002.

if (nargin<2)
   m=n;
end

W=rand(m)-.5;
[W,cococococo]=qr(W);
W=W(1:m,1:n);
