#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> 
#include <lapacke.h>
#include <complex.h>
#include <omp.h>
#include "mmio.h"

//gcc -Wall -fopenmp puissances.c -o puissances  -llapacke -lblas -lm mmio.c
double* lectureMtxFormat(char* nomfile,int M, int N, int nz){
    int ret_code;
    MM_typecode matcode;
    FILE *f;   
    int *ii, *jj;
    double *val;
    int i=0;
    if ((f = fopen(nomfile, "r")) == NULL) 
        exit(1);
    
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
        exit(1);

    ii = (int *) malloc(nz * sizeof(int));
    jj = (int *) malloc(nz * sizeof(int));
    val = (double *) malloc(nz * sizeof(double));
    
    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &ii[i], &jj[i], &val[i]);
        ii[i]--;  /* adjust from 1-based to 0-based */
        jj[i]--;
    }

    if (f !=stdin) fclose(f);


    mm_write_banner(stdout, matcode);
    mm_write_mtx_crd_size(stdout, M, N, nz);
    double* matrice =malloc(M*N*sizeof(double));
    for(i=0;i<M;i++) {
        for(int c=0;c<N;c++) {
        matrice [i*M+c]=0;
        }
    }
    for (i=0; i<nz; i++)
        matrice[ii[i]*M + jj[i]]= val[i];
    return matrice;
}

double* lectureMatrice(int m, int n){
        int i=0,j=0;
        double* A=malloc(m*n*sizeof(double));  
        for (i = 0; i < m; i++)
        {
            for (j = 0; j < n; j++)
            {
                printf("\n element [%d][%d]  :: ",i,j);
                scanf("%lf", &A[i*n+j]);
            }
        }
    return A;
}

double* genMatrice(int m, int n){
        int i=0,j=0;
        double* A=malloc(m*n*sizeof(double));
        for (i = 0; i < m; i++)
        {
            for (j = 0; j < n; j++)
            {
                A[i*n+j]= rand()%100 ;
            }
        }
    return A;
}

void afficherMatrice(int m, int n, double* A){
    int i=0,j=0;  
    for (i = 0; i < m; ++i){
        for (j = 0; j < n; ++j)
            {
                printf("\t%lf", A[i*n+j]);
            }
        printf("\n\n");
    }
}

void afficherVecteur(int n, double* V){
    for (int j = 0; j < n; ++j)
    {
        printf("\t%lf", V[j]);
        printf("\n\n");
    }
}

double dotProduct(double* v1, double* v2, int n){
    double res=0;
    for(int i=0;i<n;i++) {
        res+=v1[i]*v2[i];    
    }
    return res;
}

double normeVect(double* v, int n){
    return sqrt(dotProduct(v,v,n));
}

double* vectNormal(double* v, int n){
    double* vn= malloc(n*sizeof(double));
    int i=0;
    double norme=normeVect(v,n);
        for(i=0;i<n;i++) {
            vn[i]=v[i]/norme;    
        } 
    return vn;
}

double normeVectInf(double* v, int n){
    int i =0;
    double max=fabs(v[0]);
    i++;
    while(i<n)
    {
        if(fabs(v[i])>max) max=fabs(v[i]);
        i++;
    }
    return max;
}

double normeFob(int n, int m, double* A){   
    double res=0;
    for(int i=0;i<n*m;i++) {
        res+=A[i]*A[i];    
    }

    return sqrt(res);
}

double* multMatricesVect(double* A, double* V, int n_a, int m_a){
double* C=malloc(n_a*sizeof(double)); 
int k=0;
for(int col=0; col<n_a; col++){
    double sum = 0;
    for(int i=0; i<m_a; i++)
        {
            sum += A[m_a*col+i] * V[i];
        }   
    C[k] = sum;
    k++;
    }

return C;
}

double  puissance(double* A, double*v , int m, int n,int k ,double a,double* lambda, double** u){
double * res[k];
double * alpha=malloc(k*sizeof(double));
double * err=malloc(k*sizeof(double));
double * w=malloc(n*sizeof(double));
double norme= normeVect(v,n);
double normeA= normeFob(n,n,A);
v=vectNormal(v,n);
res[0]=multMatricesVect(A,v, n, n);
//alpha[0]=fabs(dotProduct(res[0],v,n));
alpha[0]=dotProduct(res[0],v,n);
for (int j=1;j<k;j++){
    v=res[j-1];
    v=vectNormal(v,n);
    res[j]= multMatricesVect(A,v, n, n);
    //alpha[j]=fabs(dotProduct(res[j],v,n));
    alpha[j]=dotProduct(res[j],v,n);
    norme= normeVect(res[j],n);
    for (int i = 0; i < n; i++)
        {res[j][i]= res[j][i]/norme; }
    res[j]=vectNormal(res[j],n);
    err[j]=fabs(alpha[j]-alpha[j-1])/normeA;
}
    *lambda=alpha[k-1];
    *u=res[k-1];
    FILE * fp;
    fp = fopen ("errors.txt","w");
    for(int i = 1; i < k;i++){
        fprintf (fp, "%lf\n",err[i]);
    }
    fclose (fp);
return alpha[k-1];
}

double  puissanceDeflation(double* A, double*v , int m, int n,int k ,double a, double lambda, double* u,double* lambdaS, double** uS,double** Ad){
double* v1=malloc(n*sizeof(double));
double* Adef=malloc(n*n*sizeof(double));
double * res[k];
double * alpha=malloc(k*sizeof(double));
double * err=malloc(k*sizeof(double));
double * w=malloc(n*sizeof(double));
double norme;
//norme= normeVect(u,n);
for (int i=0;i<n;i++){
        for (int j=0;j<n;j++) {
        Adef[i*n+j]=A[i*n+j]- lambda*u[i]*u[j];
        }
    }

double normeA= normeFob(n,n,Adef);
v=vectNormal(v,n);
res[0]=multMatricesVect(Adef,v, n, n);
alpha[0]=dotProduct(res[0],v,n);

for (int j=1;j<k;j++){
    v=res[j-1];
    norme= normeVect(v,n);
    v=vectNormal(v,n);
    res[j]= multMatricesVect(Adef,v, n, n);
    alpha[j]=dotProduct(res[j],v,n);
    v=vectNormal(res[j],n);
    err[j]=fabs(alpha[j]-alpha[j-1])/normeA;
}
*lambdaS=alpha[k-1];
*uS=res[k-1];
(*Ad)= Adef;
FILE * fp;
fp = fopen ("errors2.txt","w");
for(int i = 1; i < k;i++){
    fprintf (fp, "%lf\n",err[i]);
}
fclose (fp);
return alpha[k-1];
}

int main ( int argc , char * argv[] ){

double* A;
double* Ad;
double* v;
double* res;
int i=0,j=0,m,n,k;
double lamb,l2,lambda;
double* u=malloc(n*sizeof(double));
double* u2=malloc(n*sizeof(double));
/***************** MATRICE ALEATOIRE***********************/
 /*  
        printf("Entrer le nbr de lignes :: ");
        scanf("%d", &m);
        printf("\nEntrer le nbr de colonnes :: ");
        scanf("%d",&n);
        printf("\nEntrer le nbr d'itérations' :: ");
        scanf("%d",&k);

        A= genMatrice(m,n);
        v= genMatrice(1,n);
        lambda=puissance(A,v,n,m,k,1,&lamb,&u);
        printf(" Valeur propre 1 %lf \n",lamb); 
        lambda=puissanceDeflation(A,v,n,m,k,1,lamb,u,&l2,&u2,&Ad);
        printf(" Valeur propre 2 avec puissance déflaté %lf \n",lambda); 
*/
/**********************************************************/


/****************BENSHMARK DE MARTIXMARKET*****************/
/*
        n=180; m=180; k=100;
        A=malloc(180*180*sizeof(double));
        A=lectureMtxFormat("mcca.mtx",180,180,2659);
        v= genMatrice(1,n);
        double lamb,l2;
        double* u=malloc(n*sizeof(double));
        double* u2=malloc(n*sizeof(double));
        lambda=puissance(A,v,n,m,k,1,&lamb,&u);
        printf(" Valeur propre 1 %lf \n",lamb); 
        lambda=puissanceDeflation(A,v,n,m,k,1,lamb,u,&l2,&u2,&Ad);
        printf(" Valeur propre 2 avec puissance déflaté %lf \n",lambda); 
*/
/*********************************************************/

        printf(" Calcul de toutes les valeurs propres \n \n");

        /****************************************************/
        /*
        printf("Entrer le nbr de lignes :: ");
        scanf("%d", &m);
        printf("\nEntrer le nbr de colonnes :: ");
        scanf("%d",&n);
        printf("\nEntrer le nbr d'itérations' :: ");
        scanf("%d",&k);
        
        A= genMatrice(m,n);
        v= genMatrice(1,n);
        */
        /*********************************************************/
        
        n=180; m=180; k=100;
        A=malloc(180*180*sizeof(double));
        A=lectureMtxFormat("mcca.mtx",180,180,2659);
        v= genMatrice(1,n);
        
        /***********************************************************/
        int p=1;
        lambda=puissance(A,v,n,m,k,1,&lamb,&u);
        printf(" Valeur propre 1 %lf \n",lamb); 
        while(p<n && lambda!=0) {
            for (int i=0;i<n;i++){
                for (int j=0;j<n;j++) {
                    A[i*n+j]=A[i*n+j]- lambda*u[i]*u[j];
                }
            }
            lambda=puissance(A,v,n,m,k,1,&lambda,&u);
            printf(" Valeur propre %d %lf \n",p+1,lambda); 
            p++;
        } 
}

















