#! /usr/bin/octave -qf
filename = argv (){1};
kica_path = argv (){2};
addpath(kica_path);
load(filename);
m=size(x,1);
if size(x,2) > 1000
	printf('Setting recommend parameters for more than 1000 energy channels\n')
	W=kernel_ica_options(x,'contrasttype','oneunit','ncomp',m, ...
	 'W0',rand_orth(m),'contrast','kgv','kernel','hermite','sig',1.5,'kap',.0001, ...
	 'p',3,'polish',1,'restarts',4,'disp',0);
	W=kernel_ica_options(x,'W0',W,'contrast','kgv','kernel','gaussian','sig',0.5,'kap',0.001, ...
	 'p',4,'polish',0,'restarts',1,'disp',0);
else
	printf('Setting recommend parameters for less than 1000 energy channels\n')
	W=kernel_ica_options(x,'contrasttype','oneunit','ncomp',m, ...
	 'W0',rand_orth(m),'contrast','kgv','kernel','hermite','sig',1.5,'kap',.0001, ...
	 'p',3,'polish',1,'restarts',4,'disp',0);
	W=kernel_ica_options(x,'W0',W,'contrast','kgv','kernel','gaussian','sig',1,'kap',.02, ...
	 'p',4,'polish',0,'restarts',1,'disp',0);
endif

w = W
save('-v4',filename,'w');
exit;