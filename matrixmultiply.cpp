//==============================================================
//
// SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
// http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
//
// Copyright 2016-2017 Intel Corporation
//
// THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
//
// =============================================================
/*******************************************************************************
*   This example measures performance of computing the real matrix product
*   C=alpha*A*B+beta*C using a triple nested loop, where A, B, and C are
*   matrices and alpha and beta are double precision scalars.
*
*   In this simple example, practices such as memory management, data alignment,
*   and I/O that are necessary for good programming style and high MKL
*   performance are omitted to improve readability.
********************************************************************************/

#define min(x,y) (((x) < (y)) ? (x) : (y))

#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include <iostream>
#include "mkl_vsl.h"
#include <time.h>
using namespace std;

/* Consider adjusting LOOP_COUNT based on the performance of your computer */
/* to make sure that total run time is at least 1 second */
#define LOOP_COUNT 10

double *CreateMatrix(int m, int n, bool zero = false) {
	double *A = (double*)mkl_malloc(m * n * sizeof(double), 16);
	if (zero) {
		memset((void*)A, 0, m * n * sizeof(double));
	}
	else {
		int count = m * n;
		for (int i = 0; i < count; i++) {
			A[i] = (double)(i + 1);
		}
	}
	return A;
}

double *outTranspose(double *A, int m, int n) {
	double *B = CreateMatrix(m, n);
	mkl_domatcopy('R', 'T', m, n, 1.0, A, n, B, m);
	return B;
}


int main()
{
	double *A, *Uk, *X, *X_k, *X_next;
	int  n, k, i, j, r;
	double alpha, beta;
	double sum;
	double s_initial, s_elapsed;

	// random number generator - seed
	srand(time(NULL));

	printf("\n This example measures performance of rcomputing the real matrix product \n"
		" C=alpha*A*B+beta*C using a triple nested loop, where A, B, and C are \n"
		" matrices and alpha and beta are double precision scalars \n\n");

	// dimensions
	n = 3, k = 2;
	printf(" Initializing data for matrix multiplication C=A*B for matrix \n"
		" A(%ix%i) and matrix B(%ix%i)\n\n", n, n, n, k);
	alpha = 1.0; beta = 0.0;

	printf(" Allocating memory for matrices aligned on 64-byte boundary for better \n"
		" performance \n\n");
	A = CreateMatrix(n, n);
	Uk = CreateMatrix(n, k);
	//Uk_T = outTranspose(Uk, n, k);
	X = CreateMatrix(n, 1);
	X_k = CreateMatrix(k, 1);
	X_next = CreateMatrix(n, 1);

	if (A == NULL || Uk == NULL || X == NULL) {
		printf("\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
		mkl_free(A);
		mkl_free(Uk);
		mkl_free(X);
		return 1;
	}

	printf(" Making the first run of matrix product using Intel(R) MKL dgemm function \n"
		" via CBLAS interface to get stable run time measurements \n\n");


	printf(" Measuring performance of matrix product using Intel(R) MKL dgemm function \n"
		" via CBLAS interface \n\n");
	s_initial = dsecnd();
	// Uk^T * X
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		k, 1, n, alpha, Uk, k, X, 1, 0.0, X_k, 1);
	// Uk * (Uk^T * X)
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		n, 1, k, alpha, Uk, k, X_k, 1, 0.0, X_next, 1);

	// X = A * X + alpha * Uk * (Uk^T * X)
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		n, 1, n, alpha, A, n, X, 1, 1.0, X_next, 1);
	// update x 
	memcpy(X, X_next, sizeof n*1*sizeof(double));

	s_elapsed = (dsecnd() - s_initial) / LOOP_COUNT;

	printf(" == Matrix multiplication using Intel(R) MKL dgemm completed == \n"
		" == at %.5f milliseconds == \n\n", (s_elapsed * 1000));

	if (s_elapsed < 0.9 / LOOP_COUNT) {
		s_elapsed = 1.0 / LOOP_COUNT / s_elapsed;
		i = (int)(s_elapsed*LOOP_COUNT) + 1;
		printf(" It is highly recommended to define LOOP_COUNT for this example on your \n"
			" computer as %i to have total execution time about 1 second for reliability \n"
			" of measurements\n\n", i);
	}

	printf(" Top left corner of matrix A: \n");
	for (i = 0; i<n; i++) {
		for (j = 0; j<n; j++) {
			printf("%12.0f", A[j + i*n]);
		}
		printf("\n");
	}
	printf(" Top left corner of matrix Uk: \n");
	for (i = 0; i<n; i++) {
		for (j = 0; j<k; j++) {
			printf("%12.0f", Uk[j + i*k]);
		}
		printf("\n");
	}
	printf(" Top left corner of matrix X: \n");
	for (i = 0; i<n; i++) {
		for (j = 0; j<1; j++) {
			printf("%12.0f", X[j + i * 1]);
		}
		printf("\n");
	}
	printf(" Top left corner of matrix X_k: \n");
	for (i = 0; i<k; i++) {
		for (j = 0; j<1; j++) {
			printf("%12.0f", X_k[j + i * 1]);
		}
		printf("\n");
	}
	printf(" Top left corner of matrix X_next: \n");
	for (i = 0; i<n; i++) {
		for (j = 0; j<1; j++) {
			printf("%12.0f", X_next[j + i * 1]);
		}
		printf("\n");
	}

	printf(" Deallocating memory \n\n");
	mkl_free(A);
	mkl_free(Uk);
	mkl_free(X);

	printf(" Example completed. \n\n");
	cin.get();
	return 0;
}
