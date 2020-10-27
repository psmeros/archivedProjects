#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdbool.h>
#include <cuda.h>


#include "mmio_wrapper.h"
#include "util.h"
#include "parameters.h"
#include "second.h"

#include "cg_cuda.h"

/*
Implementation of a simple CUDA CG solver using matrix in the mtx format (Matrix market)
Any matrix in that format can be used to test the code
*/
int main ( int argc, char **argv ) {

	double * A = NULL;
	double * A_cuda = NULL;
	double * X_cuda = NULL;
    double * Y_cuda = NULL;
	double * x = NULL;
	double * b = NULL;

	double t1=0.;
	double t2=0.;
	double tavg=0.;

	int n = 0;
	double h;

	int nthreads=0;
	int *m_locals = NULL;
	int *A_all_pos = NULL;
	int *m_locals_cuda = NULL;
	int *A_all_pos_cuda = NULL;


	if (argc != 5){
		fprintf(stderr, "Usage: %s -M martix_market_filename -np <NUM_OF_CUDA_THREADS> \n", argv[0]); 
		exit(EXIT_FAILURE);
	}
	if (strcmp(argv[1], "-M")==0){ 		
		A = read_mat(argv[2]);
		n = get_size(argv[2]).n;
	}
	if (strcmp(argv[3], "-np")==0){
		nthreads = atoi(argv[4]);
		//Set maximum parallelization to the number of rows/columns of matrix A
		if (nthreads > n)
			nthreads=n;
	}
	printf("Matrix %s with %d number of rows/columns loaded from file \n", argv[2], n);

	h = 1./(double)n;
	b = init_source_term(n,h);
	x = (double*) calloc(n, sizeof(double));

	//Threads position on Matrix A
	m_locals = (int*) malloc(nthreads * sizeof(int));
	A_all_pos = (int*) malloc(nthreads * sizeof(int));

	for(int i=0; i<nthreads; ++i){
		
		int m_local = ceil((double) n/nthreads);
		if (i == (nthreads - 1))
			m_local = n - m_local*(nthreads-1);

		int A_pos = (int)(i * ceil((double) n/nthreads));

		m_locals[i] = m_local;
		A_all_pos[i] = A_pos;
	}

	//Allocate space and copy A matrix to the device
	init_cuda(&A_cuda, &X_cuda, &Y_cuda, &m_locals_cuda, &A_all_pos_cuda, A, m_locals, A_all_pos, n, nthreads);

	printf("Call CUDA cgsolver() on matrix size (%d x %d)\n",n,n);
	t1 = second();
	tavg = cgsolver(A_cuda, X_cuda, Y_cuda, m_locals_cuda, A_all_pos_cuda, b, x, n, nthreads);
	t2 = second();
	printf("Time for CUDA cgsolver()= %f [s]\n\n",(t2-t1));

	//Free device space
	finalize_cuda(A_cuda, X_cuda, Y_cuda, m_locals_cuda, A_all_pos_cuda);
	
	free(b);
	free(x);

	//Write measured times on file
	FILE *fp = fopen("results.csv", "a");
	fprintf(fp, "cuda, %d, %d, %f, %f\n",n,nthreads,tavg,t2-t1);
	fclose(fp);

	return 0;
}

