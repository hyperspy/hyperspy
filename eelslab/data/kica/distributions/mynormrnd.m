function r = mynormrnd(mu,sigma,m,n);
%NORMRND Random matrices from normal distribution.
%   R = NORMRND(MU,SIGMA) returns a matrix of random numbers chosen   
%   from the normal distribution with parameters MU and SIGMA.
%
%   The size of R is the common size of MU and SIGMA if both are matrices.
%   If either parameter is a scalar, the size of R is the size of the other
%   parameter. Alternatively, R = NORMRND(MU,SIGMA,M,N) returns an M by N  
%   matrix.

if nargin < 2, 
    error('Requires at least two input arguments.');
end

if nargin == 2
    [errorcode rows columns] = myrndcheck(2,2,mu,sigma);
end

if nargin == 3
    [errorcode rows columns] = myrndcheck(3,2,mu,sigma,m);
end

if nargin == 4
    [errorcode rows columns] = myrndcheck(4,2,mu,sigma,m,n);
end

if errorcode > 0
    error('Size information is inconsistent.');
end

%Initialize r to zero.
r = zeros(rows, columns);

r = randn(rows,columns) .* sigma + mu;

% Return NaN if SIGMA is not positive.
if any(any(sigma <= 0));
    if prod(size(sigma) == 1)
        tmp = NaN;
        r = tmp(ones(rows,columns));
    else
        k = find(sigma <= 0);
        tmp = NaN;
        r(k) = tmp(ones(size(k)));
    end
end
