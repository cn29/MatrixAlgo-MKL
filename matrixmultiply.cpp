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
#include <assert.h>
#include <cmath>
#include <set>
using namespace std;

/* Consider adjusting LOOP_COUNT based on the performance of your computer */
/* to make sure that total run time is at least 1 second */
#define LOOP_COUNT 5
#define ALIGN 64

#define CALL_AND_CHECK_STATUS(function, error_message) do { \
          if(function != SPARSE_STATUS_SUCCESS)             \
          {                                                 \
          printf(error_message); fflush(0);                 \
          status = 1;                                       \
          goto memory_free;                                 \
          }                                                 \
} while(0)


double *CreateMatrix(int m, int n, bool zero = false) {
	double *A = (double*)mkl_malloc(m * n * sizeof(double), ALIGN);
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


int main()
{
	double *A = NULL, *Uk, *X, *X_k, *X_next;
	int  n, k, i, j, incx, s;
	double sigma, beta;
	double s_initial, s_elapsed;
	double  *values_A = NULL;
	MKL_INT *columns_A = NULL;
	MKL_INT *rowIndex_A = NULL;
	char trans = 'n';
	// random number generator - seed
	srand(time(NULL));

	// dimensions
	n = 200000;
	k = 400;
	s = 20;
	incx = 1;
	sigma = 1.0; beta = 0.0;
	printf("\n"
		" This program compares the performance of two algorithms.\n"
		" The basic idea is: B = (A + sigma * Uk * Uk.T), X_(n+1) = B * X_(n) \n"
		" A - (%ix%i), Uk - (%ix%i), X - (%ix%i) \n"
		" Iterations s = %d, sigma = %.3f", n, n, n, k, n, 1, s, sigma);

	printf(" Allocating memory for matrices aligned on 64-byte boundary for better \n"
		" performance \n\n");
	Uk = (double *)mkl_malloc(sizeof(double)*n*k, ALIGN);
	X_next = (double *)mkl_malloc(sizeof(double)*n, ALIGN);
	X = (double *)mkl_malloc(sizeof(double)*n, ALIGN);
	X_k = (double *)mkl_malloc(sizeof(double)*k, ALIGN);
	double *y = (double *)mkl_malloc(sizeof(double) * n, ALIGN);

	// alg 2
	double *d = (double *)mkl_malloc(sizeof(double)*k, ALIGN);
	double *diagv = (double *)mkl_malloc(sizeof(double)*k, ALIGN);
	double *diagvs[30], *diagvpas[30];
	double *diagvtmp = (double *)mkl_malloc(sizeof(double)*k, ALIGN);
	double *W = (double *)mkl_malloc(sizeof(double)*k, ALIGN);
	double *W1 = (double *)mkl_malloc(sizeof(double)*k, ALIGN);
	double *diagW = (double *)mkl_malloc(sizeof(double)*k*k, ALIGN);
	double *bj = (double *)mkl_malloc(sizeof(double)*k, ALIGN);

	double density = 0.0001;
	int nnz = density * n*n;
	printf(" Density of the sparse matrix A = %0.5f, Number of non-zero element = %d\n", density, nnz);
	printf(" ***** Note: the computation of sparse matrix is neglected. ***** ");
	////// Create sparse matrix ///////
	/* Allocation of memory */
	/*
	values_A = (double *)mkl_malloc(sizeof(double) * nnz, ALIGN);
	columns_A = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * nnz, ALIGN);
	rowIndex_A = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (n + 1), ALIGN);


	int *index = (int*)mkl_malloc(nnz*sizeof(int), ALIGN);
	int ind1, ind = 0;
	set<int> indset;
	int *rowinds = (int*)mkl_malloc(sizeof(int)*n, ALIGN);
	while(indset.size() < nnz) {
	ind1 = rand() % (n*n);
	if (indset.find(ind1) == indset.end()) {
	indset.insert(ind1);
	values_A[ind++] = (double)(ind1%1000) / 288;
	}
	}
	ind = 0;
	for (set<int>::iterator iter = indset.begin(); iter != indset.end(); iter++, ind++) {
	columns_A[ind] = *iter % n;
	rowinds[*iter / n]++;
	}
	rowIndex_A[0] = 0;
	for (i = 1; i < n + 1; i++) {
	rowIndex_A[i] = rowIndex_A[i - 1] + rowinds[i - 1];
	}

	/* Printing sparse matrix */
	/*
	bool print_sparse = 0;
	if (print_sparse) {
	for (i = 0; i < nnz; i++)
	printf("%6.0f ", values_A[i]);
	printf("\n");
	for (i = 0; i < nnz; i++)
	printf("%d\t", columns_A[i]);
	printf("\n");
	for (auto iter = indset.begin(); iter != indset.end(); iter++)
	printf("%d\t", *iter);
	printf("\n");
	for (i = 0; i < n; i++)
	printf("%d\t", rowinds[i]);
	printf("\n");
	for (i = 0; i < n + 1; i++)
	printf("%d\t", rowIndex_A[i]);
	printf("\n MATRIX A:\nrow# : (value, column) (value, column)\n");
	}
	*/

	if (X_k == NULL || Uk == NULL || X == NULL || X_next == NULL) {
		printf("\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
		mkl_free(Uk);
		mkl_free(X);
		mkl_free(X_k);
		mkl_free(X_next);
		return 1;
	}

	// print X and Uk
	bool print_input = 0;
	if (print_input) {
		printf(" Sparse Matrix A: \n");
		for (int i = 0, ii = 0; i < n; i++) {
			printf("row#%d:", i + 1); fflush(0);
			for (j = rowIndex_A[i]; j < rowIndex_A[i + 1]; j++, ii++)
				printf(" (%5.0f, %6d)", values_A[ii], columns_A[ii]); fflush(0);
			printf("\n");
		}
		printf(" Matrix X: \n");
		for (i = 0; i<n; i++) {
			for (j = 0; j<1; j++)
				printf("%6.0f", X[j + i * 1]);
			printf("\n");
		}
		printf(" Matrix Uk: \n");
		for (i = 0; i<n; i++) {
			for (j = 0; j<k; j++)
				printf("%6.0f", Uk[j + i * k]);
			printf("\n");
		}
	}
	// ========== Computation Loop 1 ========== //
	printf("\n Computation of algorithm 1 .... \n");
	long long lln = n;
	long long llk = k;
	long long llone = 1;
	s_initial = dsecnd();
	for (int count = 0; count < LOOP_COUNT; count++) {
		// reset input
		for (int i = 0; i < n*k; i++)
			Uk[i] = (double)((i + 1) % 200) / 5.34e3;
		for (int i = 0; i < n * 1; i++) {
			X[i] = (double)((i + 1) % 200) / 5.34e3;
			X_next[i] = 0.0;
			y[i] = 0.0;
		}
		for (i = 0; i < (k * 1); i++)
			X_k[i] = 0.0;
		for (int ss = 0; ss < s; ss++) {
			printf("%d ", ss);
			// y = A*x
			// mkl_cspblas_dcsrgemv(&trans, &lln, values_A, rowIndex_A, columns_A, X, y);
			// UkT * X
			cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, llk, llone, lln, 1.0, Uk, llk, X, llone, 0.0, X_k, llone);
			// Uk * (Uk^T * X)
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, 1, k, sigma, Uk, k, X_k, 1, 0.0, X_next, 1);
			// X = X_next + y
			for (i = 0; i < n * 1; i++)
				X[i] = X_next[i] + y[i];
		}

		// update x 
		// memcpy(X, X_next, sizeof n * 1 * sizeof(double));
		/* Print result X */
		bool print_result = 0;
		if (print_result) {
			printf(" Result matrix X: \n");
			for (i = 0; i < n; i++) {
				for (j = 0; j < 1; j++)
					printf("%.4e", X[j + i * 1]);
				printf("\n");
			}
		}
		// double res = cblas_ddot(n, X, incx, X, incx);
		// printf("Result of vector dot product: %.5e\n", res);
		printf("Done. \n");
	}
	double s_elapsed_1 = (dsecnd() - s_initial) / LOOP_COUNT;
	printf(" == Matrix multiplication using Intel(R) MKL dgemm completed == \n"
		" == at %.5f milliseconds == \n\n", (s_elapsed_1 * 1000));



	// ========== Computation Loop 2 ========== //
	printf("\n Computation of algorithm 2 .... \n");
	s_initial = dsecnd();
	for (int count = 0; count < LOOP_COUNT; count++) {

		// -- reset inputs --
		for (int i = 0; i < n*k; i++)
			Uk[i] = (double)((i + 1) % 200) / 5.34e3;
		for (int i = 0; i < n * 1; i++) {
			X[i] = (double)((i + 1) % 200) / 5.34e3;
			X_next[i] = 0.0;
			y[i] = 0.0;
		}
		for (i = 0; i < (k * 1); i++) {
			X_k[i] = 0.0;
			d[i] = 0.0;
			diagv[i] = (double)((i + 6) % 200) / 5.34e3;
		}

		// d = UkT * X
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, llk, llone, lln, 1.0, Uk, llk, X, llone, 0.0, d, llone);
		// diagvpa = diagv + simga
		diagvs[0] = diagv;
		double *diagvpa = (double *)mkl_malloc(sizeof(double)*k, ALIGN);
		double *diagvpatmp = (double *)mkl_malloc(sizeof(double)*k, ALIGN);
		for (i = 0; i < (k * 1); i++)
			diagvpa[i] = diagv[i] + sigma;
		diagvpas[0] = diagvpa;

		for (int ss = 0; ss < s; ss++) {
			printf("%d ", ss);
			// diagv and diagvpa 
			for (int i = 0; i < k; i++) {
				diagvpatmp[i] = diagvpas[ss][i] * diagvpa[i];
				diagvtmp[i] = diagvs[ss][i] * diagv[i];
			}
			diagvpas[ss + 1] = diagvpatmp;
			diagvs[ss + 1] = diagvtmp;
			for (i = 0; i < (k * 1); i++)
				W[i] = 0.0;
			for (int i = 0; i < ss; i++) {
				for (int kk = 0; kk < k; kk++) {
					W[kk] += diagvs[i][kk] * diagvpas[ss - i + 1][kk];
				}
			}
			// W = W*sigma
			cblas_daxpy(k, sigma, W, incx, W1, incx);
			for (i = 0; i < (k * 1); i++)
				W[i] += diagvs[ss + 1][i];
			// construct a diagnal matrix 
			for (i = 0; i < (k * 1); i++)
				for (j = 0; j < (k * 1); j++)
					if (i == j)
						diagW[i*k + j] = W1[i];
					else
						diagW[i*k + j] = 0.0;
			// bj = diagW * d
			for (i = 0; i < (k * 1); i++)
				bj[i] = 0.0;
			cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, llk, llone, llk, 1.0, diagW, llk, d, llone, 0.0, bj, llone);
			// Uk * bj
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, 1, k, sigma, Uk, k, bj, 1, 0.0, X_next, 1);
			// X = X_next + y
			for (i = 0; i < n * 1; i++)
				X[i] = X_next[i] + y[i];
		}

		// double res = cblas_ddot(n, X, incx, X, incx);
		// printf("Result of vector dot product: %.5e\n", res);
		printf("Done. \n");
	}

	double s_elapsed_2 = (dsecnd() - s_initial) / LOOP_COUNT;


	printf(" == Matrix multiplication using Intel(R) MKL dgemm completed == \n"
		" == at %.5f milliseconds == \n\n", (s_elapsed_2 * 1000));

	printf(" Running time rate: %.3f\n", s_elapsed_1 / s_elapsed_2);
	printf(" Deallocating memory \n\n");
	mkl_free(A);
	mkl_free(Uk);
	mkl_free(X);
	//mkl_free(y);
	mkl_free(X_k);
	mkl_free(X_next);
	mkl_free(values_A);
	mkl_free(columns_A);
	mkl_free(rowIndex_A);

	///////////////////////////////////////////////////////////////////////////////////

	printf(" Example completed. \n\n");
	cin.get();
	return 0;
}