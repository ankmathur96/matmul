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
#define BLOCK_SIZE 128
#endif
#define INNER_BLOCK_SIZE 32

#define min(a,b) (((a)<(b))?(a):(b))

static void do_intrinsic(int lda, double* AT, double* BT, double* CT)
{
  __m256d b0 = _mm256_loadu_pd(BT);
  __m256d b1 = _mm256_loadu_pd(BT + lda);
  __m256d b2 = _mm256_loadu_pd(BT + 2*lda);
  __m256d b3 = _mm256_loadu_pd(BT + 3*lda);
  // unroll this loop.
  for (int i = 0; i < 4; i++) {
    __m256d a = _mm256_loadu_pd(AT + lda * i);
    __m256d c = _mm256_loadu_pd(CT + lda * i);
    __m256d a_elem = _mm256_set1_pd(a[0]);
    c = _mm256_fmadd_pd(a_elem, b0, c);
    a_elem = _mm256_set1_pd(a[1]);
    c = _mm256_fmadd_pd(a_elem, b1, c);
    a_elem = _mm256_set1_pd(a[2]);
    c = _mm256_fmadd_pd(a_elem, b2, c);
    a_elem = _mm256_set1_pd(a[3]);
    c = _mm256_fmadd_pd(a_elem, b3, c);
    _mm256_storeu_pd(CT + lda * i, c);
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
    double CT[4][4] = {
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
    };
    do_intrinsic(4, (double*)AT, (double*)BT, (double*)CT);
    double C[4][4];
    for (int i = 0; i < 4; i += 1)
        for (int j = 0; j < 4; j += 1)
            ((double*)C)[i+j*4] = ((double*)CT)[j+i*4];
    do_reference(4, (double*)A, (double*)B, (double*)C);
    printf("---\n");
    printf("%f %f %f %f\n", C[0][0], C[1][0], C[2][0], C[3][0]);
    printf("%f %f %f %f\n", C[0][1], C[1][1], C[2][1], C[3][1]);
    printf("%f %f %f %f\n", C[0][2], C[1][2], C[2][2], C[3][2]);
    printf("%f %f %f %f\n", C[0][3], C[1][3], C[2][3], C[3][3]);
}

static void do_block_inner_ref (int lda, int M, int N, int K, double* AT, double* BT, double* CT)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j)
    {
      /* Compute C(i,j) */
      double cij = CT[j+i*lda];
      for (int k = 0; k < K; ++k)
        cij += AT[k+i*lda] * BT[j+k*lda];
      CT[j+i*lda] = cij;
    }
}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block_inner (int lda, int M, int N, int K, double* AT, double* BT, double* CT)
{
  for (int i = 0; i < M; i += 4)
    for (int j = 0; j < N; j += 4)
      for (int k = 0; k < K; k += 4) {
        int M2 = min (4, M-i);
        int N2 = min (4, N-j);
        int K2 = min (4, K-k);
        if (M2 == 4 && N2 == 4 && K2 == 4) {
            do_intrinsic(lda, AT + k + i*lda, BT + j + k*lda, CT + j + i*lda);
        } else {
            do_block_inner_ref(lda, M2, N2, K2, AT + k + i*lda, BT + j + k*lda, CT + j + i*lda);
        }
      }
}

static void do_block (int lda, int M, int N, int K, double* AT, double* BT, double* CT)
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
          do_block_inner(lda, M2, N2, K2, AT + k + i*lda, BT + j + k*lda, CT + j + i*lda);
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
  double* AT = malloc(sizeof(double)*lda*lda);
  for (int i = 0; i < lda; i += 1)
      for (int j = 0; j < lda; j += 1)
          AT[i+j*lda] = A[j+i*lda];
  double* BT = malloc(sizeof(double)*lda*lda);
  for (int i = 0; i < lda; i += 1)
      for (int j = 0; j < lda; j += 1)
          BT[i+j*lda] = B[j+i*lda];
  double* CT = malloc(sizeof(double)*lda*lda);
  for (int i = 0; i < lda; i += 1)
      for (int j = 0; j < lda; j += 1)
          CT[i+j*lda] = C[j+i*lda];

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
        do_block(lda, M, N, K, AT + k + i*lda, BT + j + k*lda, CT + j + i*lda);
      }
  for (int i = 0; i < lda; i += 1)
      for (int j = 0; j < lda; j += 1)
          C[i+j*lda] = CT[j+i*lda];
  free(AT);
  free(BT);
  free(CT);
}
