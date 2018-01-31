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
/* Loop unrolling and double blocking works */

const char* dgemm_desc = "Simple blocked dgemm.";


#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 256
#endif
#define INNER_BLOCK_SIZE 4

#define min(a,b) (((a)<(b))?(a):(b))

#include <immintrin.h>

// // assume 4x4. 
static void do_instrinsic(double* AT, double* BT, double* CT)
{
  __m256d b0 = _mm256_load_pd(BT);
  __m256d b1 = _mm256_load_pd(BT + 4);
  __m256d b2 = _mm256_load_pd(BT + 8);
  __m256d b3 = _mm256_load_pd(BT + 12);
  // unroll this loop.
  for (int i = 0; i < 4; i++) {
    __m256d a = _mm256_load_pd(AT + i);
    __m256d c = _mm256_load_pd(CT + i);
    __m256d a_elem = _mm256_set1_pd(a[0]);
    c = _mm_fmadd_pd(a_elem, b0, c)
    __m256d a_elem = _mm256_set1_pd(a[1]);
    c = _mm_fmadd_pd(a_elem, b1, c)
    __m256d a_elem = _mm256_set1_pd(a[2]);
    c = _mm_fmadd_pd(a_elem, b2, c)
    __m256d a_elem = _mm256_set1_pd(a[3]);
    c = _mm_fmadd_pd(a_elem, b3, c)
    _mm256_store_pd(CT + i, c)
  }
}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block_inner (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */
    for (int j = 0; j < N; ++j)
    {
      /* Compute C(i,j) */
      double cij = C[j+i*lda];
      for (int k = 0; k < K; ++k)
        cij += A[k+i*lda] * B[j+k*lda];
      C[j+i*lda] = cij;
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
          if (M2 != INNER_BLOCK_SIZE || N2 != INNER_BLOCK_SIZE || K2 != INNER_BLOCK_SIZE) {
            do_block_inner(lda, M2, N2, K2, A + k + i*lda, B + j + k*lda, C + j + i*lda); 
          } else {
            do_instrinsic(lda, M2, N2, K2, A + i + k*lda, B + k + j*lda, C + i + j*lda)
          }
      }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm (int lda, double* A, double* B, double* C)
{
  double AT[lda*lda];
  for (int i = 0; i < lda; i += 1)
      for (int j = 0; j < lda; j += 1)
          AT[i+j*lda] = A[j+i*lda];
  double BT[lda*lda];
  for (int i = 0; i < lda; i += 1)
      for (int j = 0; j < lda; j += 1)
          BT[i+j*lda] = B[j+i*lda];
  double CT[lda*lda];
  for (int i = 0; i < lda; i += 1)
      for (int j = 0; j < lda; j += 1)
          CT[i+j*lda] = C[j+i*lda];
  /* For each block-row of A */
  for (int i = 0; i < lda; i += BLOCK_SIZE) {
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE) {

      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
      	/* Correct block dimensions if block "goes off edge of" the matrix */
      	int M = min (BLOCK_SIZE, lda-i);
      	int N = min (BLOCK_SIZE, lda-j);
      	int K = min (BLOCK_SIZE, lda-k);

      	/* Perform individual block dgemm */
      	do_block(lda, M, N, K, A + k + i*lda, B + j + k*lda, C + j + i*lda); 
      }
    }
  }
  // tranpose C back.
  for (int i = 0; i < lda; i += 1)
      for (int j = 0; j < lda; j += 1)
          C[i+j*lda] = CT[j+i*lda];
}




/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block_inner (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j)
    {
      /* Compute C(i,j) */
      double cij = C[i+j*lda];
      int k = 0;
      for (; k < K - 3; k += 4) {
        int offset = k * lda + i;
        double a = A[offset] * B[k+j*lda];
        double b = A[offset+lda] * B[(k+1)+j*lda];
        double c = A[offset+lda+lda] * B[(k+2)+j*lda];
        double d = A[offset+lda+lda+lda] * B[(k+3)+j*lda];
        cij += a + b + c + d;
      }
      for (; k < K; k += 1) {
        cij += A[i+k*lda] * B[k+j*lda];
      }
      C[i+j*lda] = cij;
    }
}


