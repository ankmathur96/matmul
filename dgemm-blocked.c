/* 
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 
CC = cc
OPT = -O3
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/

#include <immintrin.h>
#include <assert.h>
#include <stdio.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 256
#endif
#define INNER_BLOCK_SIZE 64

#define min(a,b) (((a)<(b))?(a):(b))

static void do_block_register (int lda, double* AT, double* B, double* C) {
  double tmp[4] __attribute__((aligned));
  __m256d a0 = _mm256_loadu_pd(AT);
  __m256d a1 = _mm256_loadu_pd(AT+lda);
  __m256d a2 = _mm256_loadu_pd(AT+2*lda);
  __m256d a3 = _mm256_loadu_pd(AT+3*lda);
  __m256d b0 = _mm256_loadu_pd(B);
  __m256d b1 = _mm256_loadu_pd(B+lda);
  __m256d b2 = _mm256_loadu_pd(B+2*lda);
  __m256d b3 = _mm256_loadu_pd(B+3*lda);
  __m256d aarr[4] = {a0, a1, a2, a3};
  __m256d barr[4] = {b0, b1, b2, b3};
  for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
          double cij = C[i + j*lda];
          _mm256_storeu_pd(tmp, _mm256_mul_pd(aarr[i], barr[j]));
          cij += tmp[0] + tmp[1] + tmp[2] + tmp[3];
          C[i + j*lda] = cij;
      }
  }
}

void do_reference (int n, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < n; ++i)
    /* For each column j of B */
    for (int j = 0; j < n; ++j) 
    {
      /* Compute C(i,j) */
      double cij = C[i+j*n];
      for( int k = 0; k < n; k++ )
        cij += A[i+k*n] * B[k+j*n];
      C[i+j*n] = cij;
    }
}

static void do_block_test() {
    double AT[4][4] = {
        {7, 2, 3, 4},
        {1, 2, 3, 4},
        {1, 2, 3, 4},
        {1, 2, 3, 4},
    };
    double A[4*4];
    for (int i = 0; i < 4; i += 1)
        for (int j = 0; j < 4; j += 1)
            A[i+j*4] = ((double*)AT)[j+i*4];
    double BT[4][4] = {
        {1, 2, 3, 4},
        {10, 2, 3, 4},
        {100, 2, 3, 4},
        {1000, 2, 3, 4},
    };
    double B[4*4];
    for (int i = 0; i < 4; i += 1)
        for (int j = 0; j < 4; j += 1)
            B[i+j*4] = ((double*)BT)[j+i*4];
    double C[4][4] = {
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
    };
    do_block_register(4, (double*)AT, (double*)B, (double*)C);
//    do_reference(4, (double*)A, (double*)B, (double*)C);
    printf("---\n");
    printf("%f %f %f %f\n", C[0][0], C[1][0], C[2][0], C[3][0]);
    printf("%f %f %f %f\n", C[0][1], C[1][1], C[2][1], C[3][1]);
    printf("%f %f %f %f\n", C[0][2], C[1][2], C[2][2], C[3][2]);
    printf("%f %f %f %f\n", C[0][3], C[1][3], C[2][3], C[3][3]);
}

static void do_block_inner_ref (int lda, int M, int N, int K, double* AT, double* B, double* C)
{
  double tmp[4] __attribute__((aligned));
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j)
    {
      /* Compute C(i,j) */
      double cij = C[i+j*lda];
      __m256d ctmp = _mm256_setzero_pd();
      int k = 0;
      for (; k < K - 3; k += 4) {
        __m256d a0 = _mm256_loadu_pd(AT + i*lda + k);
        __m256d b0 = _mm256_loadu_pd(B + j*lda + k);
        ctmp = _mm256_fmadd_pd(a0, b0, ctmp);
      }
      for (; k < K; k += 1) {
        cij += AT[k+i*lda] * B[k+j*lda];
      }
      _mm256_store_pd(tmp, ctmp);
      cij += tmp[0] + tmp[1] + tmp[2] + tmp[3];
      C[i+j*lda] = cij;
    }
}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block_inner (int lda, int M, int N, int K, double* AT, double* B, double* C)
{
  for (int i = 0; i < M; i += 4)
    for (int j = 0; j < N; j += 4)
      for (int k = 0; k < K; k += 4) {
        int M2 = min (4, M-i);
        int N2 = min (4, N-j);
        int K2 = min (4, K-k);
        if (M2 == 4 && N2 == 4 && K2 == 4) {
            do_block_register(lda, AT + k + i*lda, B + k + j*lda, C + i + j*lda);
        } else {
            do_block_inner_ref(lda, M2, N2, K2, AT + k + i*lda, B + k + j*lda, C + i + j*lda);
        }
      }
}

static void do_block (int lda, int M, int N, int K, double* AT, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < M; i += INNER_BLOCK_SIZE)
    /* For each column j of B */ 
    for (int j = 0; j < N; j += INNER_BLOCK_SIZE)
    {
      for (int k = 0; k < K; k += INNER_BLOCK_SIZE) {
          int M2 = min (INNER_BLOCK_SIZE, M-i);
          int N2 = min (INNER_BLOCK_SIZE, N-j);
          int K2 = min (INNER_BLOCK_SIZE, K-k);
          do_block_inner(lda, M2, N2, K2, AT + k + i*lda, B + k + j*lda, C + i + j*lda);
//          do_block_inner_ref(lda, M2, N2, K2, AT + k + i*lda, B + k + j*lda, C + i + j*lda);
      }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  assert(BLOCK_SIZE % 4 == 0);
  assert(INNER_BLOCK_SIZE % 4 == 0);
//  do_block_test();
  double AT[lda*lda];
  for (int i = 0; i < lda; i += 1)
      for (int j = 0; j < lda; j += 1)
          AT[i+j*lda] = A[j+i*lda];

  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M = min (BLOCK_SIZE, lda-i);
        int N = min (BLOCK_SIZE, lda-j);
        int K = min (BLOCK_SIZE, lda-k);

        /* Perform individual block dgemm */
        do_block(lda, M, N, K, AT + k + i*lda, B + k + j*lda, C + i + j*lda);
      }
}
