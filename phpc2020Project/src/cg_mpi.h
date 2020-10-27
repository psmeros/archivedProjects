#ifndef CG_MPI_H_
#define CG_MPI_H_

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <math.h>
#include <cblas.h>
#include <mpi.h>
#include "second.h"

#define MAIN_PROC 0

double cgsolver( double *A_local, double *b, double *x, int m_local, int n, int *m_locals, int *A_all_pos, const int myid, const int nprocs);

#endif /*CG_MPI_H_*/


