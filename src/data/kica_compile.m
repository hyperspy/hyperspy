#! /usr/bin/octave -qf
kica_path = argv (){1};
printf(kica_path)
cd(kica_path);
mex chol_gauss.c;
mex chol_hermite.c;
exit;
