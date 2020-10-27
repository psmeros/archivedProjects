#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdbool.h>
#include <mpi.h>


#include "mmio_wrapper.h"
#include "util.h"
#include "parameters.h"
#include "cg_mpi.h"
#include "second.h"

#define MAIN_PROC 0


/*
Implementation of a simple MPI CG solver using matrix in the mtx format (Matrix market)
Any matrix in that format can be used to test the code
*/
int main ( int argc, char **argv ) {

	double * A = NULL;
	double * x = NULL;
	double * b = NULL;

	double t1=0.;
	double t2=0.;
	double tavg = 0.;

	int n = 0;
	double h;

	//Local variables for processes
	int m_local, A_pos;
	double *A_local, *Y_local;   
	int *m_locals = NULL;
	int *A_all_pos = NULL;

	//Initilize MPI
	int myid, nprocs;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	if (myid == MAIN_PROC){

		if (argc != 3){
			fprintf(stderr, "Usage: mpiexec -np <NUM_OF_PROCESSES> %s -M martix_market_filename \n", argv[0]); 
			exit(EXIT_FAILURE);
		}
		if (strcmp(argv[1], "-M")==0){ 		
			A = read_mat(argv[2]);
			n = get_size(argv[2]).n;
		}		
		printf("Matrix %s with %d number of rows/columns loaded from file \n", argv[2], n);

		h = 1./(double)n;
		b = init_source_term(n,h);

		//Initial Split of Matrix A
		m_locals = (int*) malloc(nprocs * sizeof(int));
		A_all_pos = (int*) malloc(nprocs * sizeof(int));

		for(int i=1; i<nprocs; ++i){
			
			m_local = ceil((double) n/nprocs);
			if (i == (nprocs - 1))
				m_local = n - m_local*(nprocs-1);

			A_pos = (int)(i * ceil((double) n/nprocs));

			m_locals[i] = m_local;
			A_all_pos[i] = A_pos;

			MPI_Send(&m_local, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&n, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&(A[A_pos * n]), m_local*n, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		}

		//Last process (main) takes the leftover rows
		m_local = ceil((double) n/nprocs);
		A_local = (double*) calloc(m_local*n, sizeof(double));
		A_pos = 0;
		cblas_dcopy(m_local*n,&(A[A_pos]),1,A_local,1);
		m_locals[0] = m_local;
		A_all_pos[0] = A_pos;

		//A is distributed so it's not needed anymore
		free(A);
	}
	else{
		//Processes receiving the right partition of A
		MPI_Recv(&m_local, 1, MPI_INT, MAIN_PROC, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&n, 1, MPI_INT, MAIN_PROC, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		A_local = (double*) calloc(m_local*n, sizeof(double));
		MPI_Recv(A_local, m_local*n, MPI_DOUBLE, MAIN_PROC, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);         
   }
	
	x = (double*) calloc(n, sizeof(double));
	//Barrier to synchronize processes
	MPI_Barrier(MPI_COMM_WORLD);

	if (myid == MAIN_PROC){
		printf("Call MPI cgsolver() on matrix size (%d x %d)\n",n,n);
		t1 = second();
	}
	
	//Distributed call of cgsolver for all the processes with the proper partition of A
	tavg=cgsolver(A_local, b, x, m_local, n, m_locals, A_all_pos, myid, nprocs);
	
	if (myid == MAIN_PROC){
		t2 = second();
		printf("Time for MPI cgsolver() = %f [s]\n\n",(t2-t1));

		free(b);	
   		free(m_locals);
   		free(A_all_pos);
		
		//Write measured times on file
		FILE *fp = fopen("results.csv", "a");
		fprintf(fp, "mpi, %d, %d, %f, %f\n",n,nprocs,tavg,t2-t1);
		fclose(fp);

	}
	free(A_local);
	free(x);

	//Finalize MPI
	MPI_Finalize();
	return 0;
}


