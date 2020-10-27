extern "C" {
   #include "cg_cuda.h"
}

const double TOLERANCE = 1.0e-10;

/*
	cgsolver with CUDA support solves the linear equation A*x = b where A is of size m x n
*/

double cgsolver(double *A_cuda, double *X_cuda, double *Y_cuda, int *m_locals_cuda, int *A_all_pos_cuda, double *b, double *x, int n, int nthreads){

	double * r;
	double * rt;
	double * p;
	double * pt;
	double rsold = 0.0;
	double rsnew = 0.0;
	double * Ap;
	double * tmp;
	
	double alpha;

	int k = 0;

	r = (double*) calloc(n, sizeof(double));
	rt = (double*) calloc(n, sizeof(double));
	p = (double*) calloc(n, sizeof(double));
	pt = (double*) calloc(n, sizeof(double));
	Ap = (double*) calloc(n, sizeof(double));
	tmp = (double*) calloc(n, sizeof(double));

	double t1=0., t2=0., tavg=0.;

	//matrix-vector multiplication on the device
	mvm(A_cuda, X_cuda, Y_cuda, m_locals_cuda, A_all_pos_cuda, x, Ap, n, nthreads);

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
	
	while (k < n){
		t1 = second();
		//Ap = A * p;
		mvm(A_cuda, X_cuda, Y_cuda, m_locals_cuda, A_all_pos_cuda, p, Ap, n, nthreads);
		t2 = second();
		tavg = t2-t1;

		//alpha = rsold / (p' * Ap);
		alpha = rsold / fmax(cblas_ddot(n,pt,1,Ap,1),0.);
		//x = x + alpha * p;
		cblas_daxpy(n,alpha,p,1,x,1);
		//r = r - alpha * Ap;
		cblas_daxpy(n,-alpha,Ap,1,r,1);
		//rsnew = r' * r;
		rsnew = cblas_ddot (n,r,1,r,1);
		// Convergence test
		if ( sqrt(rsnew) < TOLERANCE ) break; 
		//p = r + (rsnew / rsold) * p;
		cblas_dcopy(n,r,1,tmp,1);	
		cblas_daxpy(n,(double)(rsnew/rsold),p,1,tmp,1);
		cblas_dcopy(n,tmp,1,p,1);
		cblas_dcopy(n,p,1,pt,1);	
		//rsold = rsnew;
		rsold = rsnew;
		k++;
	}

	printf("\t[STEP %d] residual = %E\n",k,sqrt(rsold));
	
	free(r);
	free(rt);
	free(p);
	free(pt);
	free(Ap);
	free(tmp);

	return tavg/k;
}

//perform in-thread multiplication
__global__ void mvm_gpu(double *A_cuda, double *X_cuda, double *Y_cuda, int *m_locals_cuda, int *A_all_pos_cuda, int n, int nthreads){
  int t = blockIdx.x * blockDim.x + threadIdx.x;

  if (t < nthreads){    
    for (int i=A_all_pos_cuda[t]; i<A_all_pos_cuda[t]+m_locals_cuda[t]; ++i) {
      Y_cuda[i] = 0.;
      for (int j=0; j<n; ++j)
        Y_cuda[i] += A_cuda[i * n + j] * X_cuda[j];
    }
  }
}

//Copy multiplier to device, perform multiplication, copy result from device
void mvm(double *A_cuda, double *X_cuda, double *Y_cuda, int *m_locals_cuda, int *A_all_pos_cuda, double *X, double *Y, int n, int nthreads){
  cudaMemcpy(X_cuda, X, n*sizeof(double), cudaMemcpyHostToDevice);
  mvm_gpu<<<(int)ceil((double)nthreads/(double)MAX_THREADS_PER_BLOCK),MAX_THREADS_PER_BLOCK>>>(A_cuda, X_cuda, Y_cuda, m_locals_cuda, A_all_pos_cuda, n, nthreads);
  cudaMemcpy(Y, Y_cuda, n*sizeof(double), cudaMemcpyDeviceToHost);
}

//Allocate space and copy A matrix to the device
void init_cuda(double **A_cuda, double **X_cuda, double **Y_cuda, int **m_locals_cuda, int **A_all_pos_cuda, double *A, int *m_locals, int *A_all_pos, int n, int nthreads){
  cudaMalloc(A_cuda, n*n*sizeof(double));
  cudaMalloc(X_cuda, n*sizeof(double));
  cudaMalloc(Y_cuda, n*sizeof(double));
  cudaMalloc(m_locals_cuda, nthreads*sizeof(int));
  cudaMalloc(A_all_pos_cuda, nthreads*sizeof(int));

  cudaMemcpy(*A_cuda, A, n*n*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(*m_locals_cuda, m_locals, nthreads*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(*A_all_pos_cuda, A_all_pos, nthreads*sizeof(int), cudaMemcpyHostToDevice);

  //A is copied on device so it's not needed anymore
  free(A);
  free(m_locals);
  free(A_all_pos);
}

//Free device space
void finalize_cuda(double *A_cuda, double *X_cuda, double *Y_cuda, int *m_locals_cuda, int *A_all_pos_cuda){
  cudaFree(A_cuda);
  cudaFree(X_cuda);
  cudaFree(Y_cuda);
  cudaFree(m_locals_cuda);
  cudaFree(A_all_pos_cuda);
}
