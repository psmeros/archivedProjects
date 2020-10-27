#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdbool.h>


#include "mmio_wrapper.h"
#include "util.h"
#include "parameters.h"
#include "cg_seq.h"
#include "second.h"


/*
Implementation of a simple sequential CG solver using matrix in the mtx format (Matrix market)
Any matrix in that format can be used to test the code
*/
int main ( int argc, char **argv ) {

	double * A = NULL;
	double * x;
	double * b;

	double t1,t2,tavg;

	int n = 0;
	double h;

	if (argc != 3){
		fprintf(stderr, "Usage: %s -M martix_market_filename \n", argv[0]); 
		exit(EXIT_FAILURE);
	}
	if (strcmp(argv[1], "-M")==0){ 		
		A = read_mat(argv[2]);
		n = get_size(argv[2]).n;
	}		
	printf("Matrix %s with %d number of rows/columns loaded from file \n", argv[2], n);

	h = 1./(double)n;
	b = init_source_term(n,h);

	x = (double*) malloc(n * sizeof(double));

	printf("Call sequential cgsolver() on matrix size (%d x %d)\n",n,n);
	t1 = second();
	tavg=cgsolver( A, b, x, n, n );
	t2 = second();
	printf("Time for sequential cgsolver()= %f [s]\n\n",(t2-t1));

	printf("AVG time for matrix-vector product calculation = %f [s]\n", tavg);

	//Write measured times on file
	FILE *fp = fopen("results.csv", "a");
	fprintf(fp, "seq, %d, 0, %f, %f\n",n,tavg,t2-t1);
	fclose(fp);

	free(A);
	free(b);
	free(x);
	return 0;
}


