function [G, Pvec] = chol_poly(x,r,s,d,tol)

% CHOL_INC_FUN - incomplete Cholesky decomposition of the Gram matrix defined
%                by data x, with the polynomial kernel with parameter r,s,d
%                Symmetric pivoting is used and the algorithms stops 
%                when the sum of the remaining pivots is less than TOL.
% 

% CHOL_INC returns returns an uPvecer triangular matrix G and a permutation 
% matrix P such that P'*A*P=G*G'.

% P is ONLY stored as a reordering vector PVEC such that 
%                    A(Pvec,Pvec)= G*G' 
% consequently, to find a matrix R such that A=R*R', you should do
% [a,Pvec]=sort(Pvec); R=G(Pvec,:);

% Copyright (c) Francis R. Bach, 2002.

n=size(x,2);
Pvec= 1:n;
I = [];
%calculates diagonal elements (not all equal to 1 for non gaussian kernels)
diagG=(r+s*sum(x.*x,1)).^d;
diagG=diagG';
i=1;
G=[];
diagK=diagG;
while ((sum(diagG(i:n))>tol)) 
   G=[G zeros(n,1)];
   % find best new element
   if i>1
      [diagmax,jast]=max(diagG(i:n));
      jast=jast+i-1;
  %    jast=i;
      %updates permutation
      Pvec( [i jast] ) = Pvec( [jast i] );
      % updates all elements of G due to new permutation
      G([i jast],1:i)=G([ jast i],1:i);
      % do the cholesky update
      
      
   else
      jast=1;
   end
   
   
   
   G(i,i)=diagG(jast); %A(Pvec(i),Pvec(i));
   G(i,i)=sqrt(G(i,i));
   if (i<n)
      %calculates newAcol=A(Pvec((i+1):n),Pvec(i))
      newAcol = (r+s*x(:,Pvec((i+1):n))'*x(:,Pvec(i)) ).^d;
      
      if (i>1)
         G((i+1):n,i)=1/G(i,i)*( newAcol - G((i+1):n,1:(i-1))*(G(i,1:(i-1)))');
      else
         G((i+1):n,i)=1/G(i,i)*newAcol;
      end
      
   end
   
   % updates diagonal elements
   if (i<n) 
      diagG((i+1):n)=diagK(Pvec((i+1):n))-sum(   G((i+1):n,1:i).^2,2  );
   end
   i=i+1;
end
