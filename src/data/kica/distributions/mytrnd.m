function r = mytrnd(v,m,n)
%TRND   Random matrices from Student's T distribution.
%   R = TRND(V) returns a matrix of random numbers chosen   
%   from the T distribution with V degrees of freedom.
%   The size of R is the size of V.
%   Alternatively, R = TRND(V,M,N) returns an M by N matrix. 

%   References:
%      [1]  L. Devroye, "Non-Uniform Random Variate Generation", 
%      Springer-Verlag, 1986


if nargin < 1, 
    error('Requires one input argument.'); 
end

if nargin == 1
    [errorcode rows columns] = mymyrndcheck(1,1,v);
end

if nargin == 2
    [errorcode rows columns] = myrndcheck(2,1,v,m);
end

if nargin == 3
    [errorcode rows columns] = myrndcheck(3,1,v,m,n);
end

if errorcode > 0
    error('Size information is inconsistent.');
end

%Initialize r to zero.
r = zeros(rows, columns);
v
whos v
if prod(double(size(v) == 1))
    v = v(ones(rows,columns));
end

% When V = 1 the T and Cauchy distributions are the same.
% Generate Cauchy random numbers as a ratio of normal. 
% Devroye p. 451.
k1 = find(v == 1);
if any(k1)
    u1 = randn(size(k1));
    u2 = randn(size(k1));
    r(k1) = u1 ./ u2;
end

k2 = find(v == 2);
% When V = 2, express T random numbers as a function of uniform
% random numbers.  See Devroye, page 430, Theorem 4.1, part C.
if any(k2)
    u = rand(size(k2));
    r(k2) = sqrt(2.0) * (u - 0.5) ./ sqrt(u - u .^ 2);
end

k = find(~(v == 2 | v == 1));
% Otherwise, express t random numbers as a function of symmetric
% beta random numbers. See Devroye, page 446, number 3.
if any(k)
    x = mybetarnd(v(k) ./ 2,v(k) ./ 2);
    r(k) = sqrt(v(k)) .* (x - 0.5) ./ sqrt(x .* (1 - x));
end

% Return NaN for values of V that are not positive integers.
if any(any(v <= 0)) | any(any(v ~= round(v)));
    if prod(size(v) == 1)
        tmp = NaN;
        r = tmp(ones(rows,columns));
    else
        k = find(v <= 0);
        tmp = NaN;
        r(k) = tmp(ones(size(k)));
    end
end
