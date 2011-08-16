#include "mex.h"
#include <math.h>
    
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
double *z,a,b,c,maxdiagG;
double sigma,tol,*temp,*diagG,*G, *Gbis;
int m, n,i,j,jast;
int iter;
int *pp;
int nmax;
double *x, *y, residual;

m = mxGetM(prhs[0]); /* dimension of input space might be greater than 1*/
n = mxGetN(prhs[0]); /* number of samples */
x = mxGetPr(prhs[0]); 
temp=mxGetPr(prhs[1]);
sigma=*temp;
temp=mxGetPr(prhs[2]);
tol=*temp;
if (nrhs>3)
	{
	temp=mxGetPr(prhs[3]);
	nmax=*temp;
	if (nmax==0) nmax=20*3*m/2; else nmax+=1+nmax/8;
	}
	else nmax=20*3*m/2; 

diagG= (double*) calloc (n,sizeof(double));
G= (double*) calloc (nmax*n,sizeof(double));
pp= (int*) calloc (n,sizeof(int));


iter=0;
residual=n;
for (i=0;i<=n-1;i++)  pp[i]=i;
for (i=0;i<=n-1;i++)  diagG[i]=1;

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
a=-.5/sigma/sigma;

for (i=iter+1; i<=n-1; i++) 
	{
	if (m<=1)
		b=(x[pp[iter]]-x[pp[i]])*(x[pp[iter]]-x[pp[i]]);
	else
		{
		b=0.0;
		for (j=0;j<=m-1;j++)
			{
			c=x[j+m*pp[iter]]-x[j+m*pp[i]];
			b+=c*c;
			}
		}
	G[i+n*iter]=exp(a*b);
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
	b=1.0;
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
free(G);
free(pp);
}


