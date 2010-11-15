#include "mex.h"
#include <math.h>
    

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
double *z,a,b,c,maxdiagG;
double sigma,tol,*temp,*diagG,*hermiteVal, *diagK, *G, *Gbis;
int m, d,n,i,j,jast;
int iter;
int *pp;
int nmax,p;
double cons,*x, *y, residual;

m = mxGetM(prhs[0]); /* dimension of input space might be greater than 1*/
n = mxGetN(prhs[0]); /* number of samples */
x = mxGetPr(prhs[0]); 
temp=mxGetPr(prhs[1]);
sigma=*temp;
temp=mxGetPr(prhs[2]);
p=*temp;
temp=mxGetPr(prhs[3]);
tol=*temp;

if (nrhs>4)
	{
	temp=mxGetPr(prhs[4]);
	nmax=*temp;
	if (nmax==0) nmax=20*3*m/2; else nmax+=1+nmax/8;
	}
	else nmax=20*3*m/2; 


diagG= (double*) calloc (n,sizeof(double));
diagK= (double*) calloc (n,sizeof(double));
G= (double*) calloc (nmax*n,sizeof(double));
hermiteVal= (double*) calloc ((p+1)*n,sizeof(double));



/* calculate Hermite values */
for (i=0;i<=n-1;i++)  
	{
	hermiteVal[i+0*n]=exp(-x[i]*x[i]/2.0/sigma/sigma);
	hermiteVal[i+1*n]=2.0*x[i]*exp(-x[i]*x[i]/2.0/sigma/sigma);
	}
for (j=2;j<=p;j++)
	for (i=0;i<=n-1;i++) 
		hermiteVal[i+j*n]=2.0*x[i]*hermiteVal[i+(j-1)*n]-2.0*(0.0+j-1)*hermiteVal[i+(j-2)*n];
pp= (int*) calloc (n,sizeof(int));



/* calculate diagonal elements */
for (i=0;i<=n-1;i++)  diagG[i]=0.0;
cons=1.0;
for (j=0;j<=p;j++)
	{
	if (j>0) cons*=2.0*j;
	for (i=0;i<=n-1;i++) 
		diagG[i]+=hermiteVal[i+j*n]*hermiteVal[i+j*n]/cons;
	}
for (i=0;i<=n-1;i++)  diagK[i]=diagG[i];


iter=0;
residual=n;
for (i=0;i<=n-1;i++)  pp[i]=i;

jast=0;

while ( residual > tol)
{

if (iter==(nmax-1))
	{
	/* need to reallocate memory to G */
	nmax+=nmax/2;
      Gbis= (double*) calloc (nmax*n,sizeof(double));
	for (i=0;i<iter*n;i++) Gbis[i]=G[i];
	free(G);
	G=Gbis;
	}


/* switches already calculated elements of G and order in pp */
if (jast!=iter)
	{
	i=pp[jast];  pp[jast]=pp[iter];  pp[iter]=i;
	for (i=0;i<=iter;i++)
		{
		a=G[jast+n*i];  G[jast+n*i]=G[iter+n*i];  G[iter+n*i]=a;
		}
	}


G[iter*(n+1)]=sqrt(diagG[jast]);

for (i=iter+1; i<=n-1; i++) 
	{	
	G[i+n*iter]=0.0;
	cons=1.0;
	for (j=0;j<=p;j++)
		{
		if (j>0) cons*=2.0*j;
	      G[i+n*iter]+=hermiteVal[pp[i]+j*n]*hermiteVal[pp[iter]+j*n]/cons;
		}
	}

if (iter>0)
	for (j=0; j<=iter-1; j++)
		for (i=iter+1; i<=n-1; i++) G[i+n*iter]-=G[i+n*j]*G[iter+n*j];

for (i=iter+1; i<=n-1; i++) 
	{
	G[i+n*iter]/=G[iter*(n+1)];

	}
residual=0.0;
jast=iter+1;
maxdiagG=0;
for (i=iter+1; i<=n-1; i++)
	{
	b=diagK[pp[i]];
	for (j=0;j<=iter;j++)
		{
		 b-=G[i+j*n]*G[i+j*n];
		}
      diagG[i]=b;
	if (b>maxdiagG)
		{
		jast=i;
		maxdiagG=b;
		}
      residual+=b;
	} 

iter++;
}

plhs[0]=mxCreateDoubleMatrix(n,iter,0); 
z= mxGetPr(plhs[0]); 
for (i=0;i<=n*iter-1;i++) z[i]=G[i];


plhs[1]=mxCreateDoubleMatrix(1,n,0); 
z= mxGetPr(plhs[1]); 
for (i=0;i<=n-1;i++) z[i]=0.0+pp[i];



free(diagG);
free(diagK);
free(G);
free(pp);
free(hermiteVal);
}


