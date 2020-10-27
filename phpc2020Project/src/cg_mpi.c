#include "cg_mpi.h"

const double TOLERANCE = 1.0e-10;

/*
	MPI in-process cgsolver solves the linear equation A*x = b where A_local is of size m_local x n (A_local is a partition of the original matrix A)
*/
double cgsolver( double *A_local, double *b, double *x, int m_local, int n, int *m_locals, int *A_all_pos, const int myid, const int nprocs){

	double * r;
	double * rt;
	double * p;
	double * pt;
	double rsold = 0.0;
	double rsnew = 0.0;
	double * Ap;
	double * tmp;
	
	double * Y_local;
	double alpha;

	int k = 0;
	int convergence = 0;

	r = (double*) calloc(n, sizeof(double));
	rt = (double*) calloc(n, sizeof(double));
	p = (double*) calloc(n, sizeof(double));
	pt = (double*) calloc(n, sizeof(double));
	Ap = (double*) calloc(n, sizeof(double));
	tmp = (double*) calloc(n, sizeof(double));

	Y_local = (double*) calloc(m_local, sizeof(double));

	double t1=0., t2=0., tavg=0.;

	//Compute the in-process matrix-vector product
	cblas_dgemv (CblasRowMajor, CblasNoTrans, m_local, n, 1., A_local, n, x, 1, 0., Y_local, 1);
	//Gather all parts of the resulting vector in the main process
	MPI_Gatherv(Y_local, m_local, MPI_DOUBLE, Ap, m_locals, A_all_pos, MPI_DOUBLE, MAIN_PROC, MPI_COMM_WORLD);   

	if (myid == MAIN_PROC){
		//r = b - A * x;
		cblas_dcopy(n,b,1,tmp,1);	
		cblas_daxpy(n,-1.,Ap,1,tmp,1);
		cblas_dcopy(n,tmp,1,r,1);
		//p = r;
		cblas_dcopy (n, r, 1, p, 1);
		cblas_dcopy(n,r,1,rt,1);	
		cblas_dcopy(n,p,1,pt,1);	
		//rsold = r' * r;
		rsold = cblas_ddot (n,r,1,rt,1);
	}

	//Convergence is checked by the main process
	while (convergence == 0){
		if (myid == MAIN_PROC)
			t1 = second();

		//Ap = A * p;
		//Broadcast the vector needed for the product
		MPI_Bcast(p, n, MPI_DOUBLE, MAIN_PROC, MPI_COMM_WORLD);
		//Compute the in-process matrix-vector product
		cblas_dgemv (CblasRowMajor, CblasNoTrans, m_local, n, 1., A_local, n, p, 1, 0., Y_local, 1);
		//Gather all parts of the resulting vector in the main process
   		MPI_Gatherv(Y_local, m_local, MPI_DOUBLE, Ap, m_locals, A_all_pos, MPI_DOUBLE, MAIN_PROC, MPI_COMM_WORLD);   
		
		if (myid == MAIN_PROC){
			t2 = second();
			tavg = t2-t1;
			//alpha = rsold / (p' * Ap);
			alpha = rsold / fmax(cblas_ddot(n,pt,1,Ap,1), 0.);
			//x = x + alpha * p;
			cblas_daxpy(n,alpha,p,1,x,1);
			//r = r - alpha * Ap;
			cblas_daxpy(n,-alpha,Ap,1,r,1);
			//rsnew = r' * r;
			rsnew = cblas_ddot (n,r,1,r,1);
			// Convergence test
			if (sqrt(rsnew) < TOLERANCE || k > n)
				convergence = 1;
			//p = r + (rsnew / rsold) * p;
			cblas_dcopy(n,r,1,tmp,1);	
			cblas_daxpy(n,(double)(rsnew/rsold),p,1,tmp,1);
			cblas_dcopy(n,tmp,1,p,1);
			cblas_dcopy(n,p,1,pt,1);	
			//rsold = rsnew;
			rsold = rsnew;
			k++;
		}
		// Broadcast convergence value to finish the loop for all the processes
		MPI_Bcast(&convergence, 1, MPI_INT, MAIN_PROC, MPI_COMM_WORLD);
	}

	if (myid == MAIN_PROC){
		printf("\t[STEP %d] residual = %E\n",k-1,sqrt(rsold));
	}
	free(r);
	free(rt);
	free(p);
	free(pt);
	free(Ap);
	free(tmp);
	free(Y_local);
	
	return tavg/k;
}


