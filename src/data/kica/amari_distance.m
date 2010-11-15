function Perf=amari_distance(Q1,Q2);
% AMARI_DISTANCE - distance between two matrices. Beware: it does not verify the axioms
%                  of a distance. It is always between 0 and 1.

% Copyright (c) Francis R. Bach, 2002.

m=size(Q1,1);
Per=inv(Q1)*Q2;
Perf=[sum((sum(abs(Per))./max(abs(Per))-1)/(m-1))/m; sum((sum(abs(Per'))./max(abs(Per'))-1)/(m-1))/m];
Perf=mean(Perf);

   
   