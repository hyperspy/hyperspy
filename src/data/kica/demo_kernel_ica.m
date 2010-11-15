% demonstration script for the kernel-ica package
fprintf('Demonstration and test of the kernel-ica package\n');
N=1000;       					%number of training samples
m=3;			  					%number of components


s=[];
for i=1:m
   switch i
   case 1, news=rand(1,N);
   case 2, news=sin((1:N)/N*20);
   case 3, news=exprnd(1,1,N);

   end
   news=news-mean(news);   % centers data
   news=news/std(news,1);  % scales data
   s=[s; news];
end

Wg=rand_orth(m);
x=Wg*s;                      % rotates data to generate mixtures

% ONE UNIT CONTRAST FUNCTION - HERMITE KERNEL  
% Wcca=kernel_ica(x,'contrasttype','oneunit','ncomp',m,'contrast','kgv', ...
%  'kernel','hermite','p',3, 'sig',1.5,'kap',.0001,'polish',1,'restarts',2);

% ONE UNIT CONTRAST FUNCTION - HERMITE KERNEL  
Wcca=kernel_ica(x);


sestimate=Wcca*x;

for i=1:m
subplot(3,m,i)
plot(s(i,:))
title(sprintf('source %d',i));
axis off;

subplot(3,m,i+m)
plot(x(i,:))
title(sprintf('mixture %d',i));
axis off;

subplot(3,m,i+2*m)
plot(sestimate(i,:))
title(sprintf('estimated source %d',i));
axis off;

end

