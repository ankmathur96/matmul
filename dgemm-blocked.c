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
#define INNER_BLOCK_SIZE 128

#define min(a,b) (((a)<(b))?(a):(b))

static void do_intrinsic(int lda, double* A, double* B, double* C)
{
  // Load all the columns of A
  __m256d a_col00 = _mm256_loadu_pd(A);
  __m256d a_col01 = _mm256_loadu_pd(A + 4);
  __m256d a_col10 = _mm256_loadu_pd(A + lda);
  __m256d a_col11 = _mm256_loadu_pd(A + lda + 4);
  __m256d a_col20 = _mm256_loadu_pd(A + 2*lda);
  __m256d a_col21 = _mm256_loadu_pd(A + 2*lda + 4);
  __m256d a_col30 = _mm256_loadu_pd(A + 3*lda);
  __m256d a_col31 = _mm256_loadu_pd(A + 3*lda + 4);
  __m256d a_col40 = _mm256_loadu_pd(A + 4*lda);
  __m256d a_col41 = _mm256_loadu_pd(A + 4*lda + 4);
  __m256d a_col50 = _mm256_loadu_pd(A + 5*lda);
  __m256d a_col51 = _mm256_loadu_pd(A + 5*lda + 4);
  __m256d a_col60 = _mm256_loadu_pd(A + 6*lda);
  __m256d a_col61 = _mm256_loadu_pd(A + 6*lda + 4);
  __m256d a_col70 = _mm256_loadu_pd(A + 7*lda);
  __m256d a_col71 = _mm256_loadu_pd(A + 7*lda + 4);

  // For each column of B + C
  /********UNROLLED*********/
  __m256d b_col0_0 = _mm256_loadu_pd(B + lda * 0);
  __m256d bcol1_0 = _mm256_loadu_pd(B + lda * 0 + 4);
  // Load the C column in 2 pieces
  __m256d c_col_0 = _mm256_loadu_pd(C + lda * 0);
  __m256d c_col1_0 = _mm256_loadu_pd(C + lda * 0 + 4);

  // Broadcast the first element of the B col, dot with A col
  __m256d b_elem_0 = _mm256_set1_pd(b_col0_0[0]);
  c_col_0 = _mm256_fmadd_pd(a_col00, b_elem_0, c_col_0);
  // Repeat...
  c_col1_0 = _mm256_fmadd_pd(a_col01, b_elem_0, c_col1_0);
  b_elem_0 = _mm256_set1_pd(b_col0_0[1]);
  c_col_0 = _mm256_fmadd_pd(a_col10, b_elem_0, c_col_0);
  c_col1_0 = _mm256_fmadd_pd(a_col11, b_elem_0, c_col1_0);
  b_elem_0 = _mm256_set1_pd(b_col0_0[2]);
  c_col_0 = _mm256_fmadd_pd(a_col20, b_elem_0, c_col_0);
  c_col1_0 = _mm256_fmadd_pd(a_col21, b_elem_0, c_col1_0);
  b_elem_0 = _mm256_set1_pd(b_col0_0[3]);
  c_col_0 = _mm256_fmadd_pd(a_col30, b_elem_0, c_col_0);
  c_col1_0 = _mm256_fmadd_pd(a_col31, b_elem_0, c_col1_0);

  // Repeat more with second half of B col
  b_elem_0 = _mm256_set1_pd(bcol1_0[0]);
  c_col_0 = _mm256_fmadd_pd(a_col40, b_elem_0, c_col_0);
  c_col1_0 = _mm256_fmadd_pd(a_col41, b_elem_0, c_col1_0);
  b_elem_0 = _mm256_set1_pd(bcol1_0[1]);
  c_col_0 = _mm256_fmadd_pd(a_col50, b_elem_0, c_col_0);
  c_col1_0 = _mm256_fmadd_pd(a_col51, b_elem_0, c_col1_0);
  b_elem_0 = _mm256_set1_pd(bcol1_0[2]);
  c_col_0 = _mm256_fmadd_pd(a_col60, b_elem_0, c_col_0);
  c_col1_0 = _mm256_fmadd_pd(a_col61, b_elem_0, c_col1_0);
  b_elem_0 = _mm256_set1_pd(bcol1_0[3]);
  c_col_0 = _mm256_fmadd_pd(a_col70, b_elem_0, c_col_0);
  c_col1_0 = _mm256_fmadd_pd(a_col71, b_elem_0, c_col1_0);

  // Store two halves of C col
  _mm256_storeu_pd(C + lda * 0, c_col_0);
  _mm256_storeu_pd(C + lda * 0 + 4, c_col1_0);

  // 1

  __m256d b_col0_1 = _mm256_loadu_pd(B + lda * 1);
  __m256d bcol1_1 = _mm256_loadu_pd(B + lda * 1 + 4);
  // Load the C column in 2 pieces
  __m256d c_col_1 = _mm256_loadu_pd(C + lda * 1);
  __m256d c_col1_1 = _mm256_loadu_pd(C + lda * 1 + 4);

  // Broadcast the first element of the B col, dot with A col
  __m256d b_elem_1 = _mm256_set1_pd(b_col0_1[0]);
  c_col_1 = _mm256_fmadd_pd(a_col00, b_elem_1, c_col_1);
  // Repeat...
  c_col1_1 = _mm256_fmadd_pd(a_col01, b_elem_1, c_col1_1);
  b_elem_1 = _mm256_set1_pd(b_col0_1[1]);
  c_col_1 = _mm256_fmadd_pd(a_col10, b_elem_1, c_col_1);
  c_col1_1 = _mm256_fmadd_pd(a_col11, b_elem_1, c_col1_1);
  b_elem_1 = _mm256_set1_pd(b_col0_1[2]);
  c_col_1 = _mm256_fmadd_pd(a_col20, b_elem_1, c_col_1);
  c_col1_1 = _mm256_fmadd_pd(a_col21, b_elem_1, c_col1_1);
  b_elem_1 = _mm256_set1_pd(b_col0_1[3]);
  c_col_1 = _mm256_fmadd_pd(a_col30, b_elem_1, c_col_1);
  c_col1_1 = _mm256_fmadd_pd(a_col31, b_elem_1, c_col1_1);

  // Repeat more with second half of B col
  b_elem_1 = _mm256_set1_pd(bcol1_1[0]);
  c_col_1 = _mm256_fmadd_pd(a_col40, b_elem_1, c_col_1);
  c_col1_1 = _mm256_fmadd_pd(a_col41, b_elem_1, c_col1_1);
  b_elem_1 = _mm256_set1_pd(bcol1_1[1]);
  c_col_1 = _mm256_fmadd_pd(a_col50, b_elem_1, c_col_1);
  c_col1_1 = _mm256_fmadd_pd(a_col51, b_elem_1, c_col1_1);
  b_elem_1 = _mm256_set1_pd(bcol1_1[2]);
  c_col_1 = _mm256_fmadd_pd(a_col60, b_elem_1, c_col_1);
  c_col1_1 = _mm256_fmadd_pd(a_col61, b_elem_1, c_col1_1);
  b_elem_1 = _mm256_set1_pd(bcol1_1[3]);
  c_col_1 = _mm256_fmadd_pd(a_col70, b_elem_1, c_col_1);
  c_col1_1 = _mm256_fmadd_pd(a_col71, b_elem_1, c_col1_1);

  // Store two halves of C col
  _mm256_storeu_pd(C + lda * 1, c_col_1);
  _mm256_storeu_pd(C + lda * 1 + 4, c_col1_1);

  // 2

  __m256d b_col0_2 = _mm256_loadu_pd(B + lda * 2);
  __m256d bcol1_2 = _mm256_loadu_pd(B + lda * 2 + 4);
  // Load the C column in 2 pieces
  __m256d c_col_2 = _mm256_loadu_pd(C + lda * 2);
  __m256d c_col1_2 = _mm256_loadu_pd(C + lda * 2 + 4);

  // Broadcast the first element of the B col, dot with A col
  __m256d b_elem_2 = _mm256_set1_pd(b_col0_2[0]);
  c_col_2 = _mm256_fmadd_pd(a_col00, b_elem_2, c_col_2);
  // Repeat...
  c_col1_2 = _mm256_fmadd_pd(a_col01, b_elem_2, c_col1_2);
  b_elem_2 = _mm256_set1_pd(b_col0_2[1]);
  c_col_2 = _mm256_fmadd_pd(a_col10, b_elem_2, c_col_2);
  c_col1_2 = _mm256_fmadd_pd(a_col11, b_elem_2, c_col1_2);
  b_elem_2 = _mm256_set1_pd(b_col0_2[2]);
  c_col_2 = _mm256_fmadd_pd(a_col20, b_elem_2, c_col_2);
  c_col1_2 = _mm256_fmadd_pd(a_col21, b_elem_2, c_col1_2);
  b_elem_2 = _mm256_set1_pd(b_col0_2[3]);
  c_col_2 = _mm256_fmadd_pd(a_col30, b_elem_2, c_col_2);
  c_col1_2 = _mm256_fmadd_pd(a_col31, b_elem_2, c_col1_2);

  // Repeat more with second half of B col
  b_elem_2 = _mm256_set1_pd(bcol1_2[0]);
  c_col_2 = _mm256_fmadd_pd(a_col40, b_elem_2, c_col_2);
  c_col1_2 = _mm256_fmadd_pd(a_col41, b_elem_2, c_col1_2);
  b_elem_2 = _mm256_set1_pd(bcol1_2[1]);
  c_col_2 = _mm256_fmadd_pd(a_col50, b_elem_2, c_col_2);
  c_col1_2 = _mm256_fmadd_pd(a_col51, b_elem_2, c_col1_2);
  b_elem_2 = _mm256_set1_pd(bcol1_2[2]);
  c_col_2 = _mm256_fmadd_pd(a_col60, b_elem_2, c_col_2);
  c_col1_2 = _mm256_fmadd_pd(a_col61, b_elem_2, c_col1_2);
  b_elem_2 = _mm256_set1_pd(bcol1_2[3]);
  c_col_2 = _mm256_fmadd_pd(a_col70, b_elem_2, c_col_2);
  c_col1_2 = _mm256_fmadd_pd(a_col71, b_elem_2, c_col1_2);

  // Store two halves of C col
  _mm256_storeu_pd(C + lda * 2, c_col_2);
  _mm256_storeu_pd(C + lda * 2 + 4, c_col1_2);

  //3

  __m256d b_col0_3 = _mm256_loadu_pd(B + lda * 3);
  __m256d bcol1_3 = _mm256_loadu_pd(B + lda * 3 + 4);
  // Load the C column in 2 pieces
  __m256d c_col_3 = _mm256_loadu_pd(C + lda * 3);
  __m256d c_col1_3 = _mm256_loadu_pd(C + lda * 3 + 4);

  // Broadcast the first element of the B col, dot with A col
  __m256d b_elem_3 = _mm256_set1_pd(b_col0_3[0]);
  c_col_3 = _mm256_fmadd_pd(a_col00, b_elem_3, c_col_3);
  // Repeat...
  c_col1_3 = _mm256_fmadd_pd(a_col01, b_elem_3, c_col1_3);
  b_elem_3 = _mm256_set1_pd(b_col0_3[1]);
  c_col_3 = _mm256_fmadd_pd(a_col10, b_elem_3, c_col_3);
  c_col1_3 = _mm256_fmadd_pd(a_col11, b_elem_3, c_col1_3);
  b_elem_3 = _mm256_set1_pd(b_col0_3[2]);
  c_col_3 = _mm256_fmadd_pd(a_col20, b_elem_3, c_col_3);
  c_col1_3 = _mm256_fmadd_pd(a_col21, b_elem_3, c_col1_3);
  b_elem_3 = _mm256_set1_pd(b_col0_3[3]);
  c_col_3 = _mm256_fmadd_pd(a_col30, b_elem_3, c_col_3);
  c_col1_3 = _mm256_fmadd_pd(a_col31, b_elem_3, c_col1_3);

  // Repeat more with second half of B col
  b_elem_3 = _mm256_set1_pd(bcol1_3[0]);
  c_col_3 = _mm256_fmadd_pd(a_col40, b_elem_3, c_col_3);
  c_col1_3 = _mm256_fmadd_pd(a_col41, b_elem_3, c_col1_3);
  b_elem_3 = _mm256_set1_pd(bcol1_3[1]);
  c_col_3 = _mm256_fmadd_pd(a_col50, b_elem_3, c_col_3);
  c_col1_3 = _mm256_fmadd_pd(a_col51, b_elem_3, c_col1_3);
  b_elem_3 = _mm256_set1_pd(bcol1_3[2]);
  c_col_3 = _mm256_fmadd_pd(a_col60, b_elem_3, c_col_3);
  c_col1_3 = _mm256_fmadd_pd(a_col61, b_elem_3, c_col1_3);
  b_elem_3 = _mm256_set1_pd(bcol1_3[3]);
  c_col_3 = _mm256_fmadd_pd(a_col70, b_elem_3, c_col_3);
  c_col1_3 = _mm256_fmadd_pd(a_col71, b_elem_3, c_col1_3);

  // Store two halves of C col
  _mm256_storeu_pd(C + lda * 3, c_col_3);
  _mm256_storeu_pd(C + lda * 3 + 4, c_col1_3);

  //4

  __m256d b_col0_4 = _mm256_loadu_pd(B + lda * 4);
  __m256d bcol1_4 = _mm256_loadu_pd(B + lda * 4 + 4);
  // Load the C column in 2 pieces
  __m256d c_col_4 = _mm256_loadu_pd(C + lda * 4);
  __m256d c_col1_4 = _mm256_loadu_pd(C + lda * 4 + 4);

  // Broadcast the first element of the B col, dot with A col
  __m256d b_elem_4 = _mm256_set1_pd(b_col0_4[0]);
  c_col_4 = _mm256_fmadd_pd(a_col00, b_elem_4, c_col_4);
  // Repeat...
  c_col1_4 = _mm256_fmadd_pd(a_col01, b_elem_4, c_col1_4);
  b_elem_4 = _mm256_set1_pd(b_col0_4[1]);
  c_col_4 = _mm256_fmadd_pd(a_col10, b_elem_4, c_col_4);
  c_col1_4 = _mm256_fmadd_pd(a_col11, b_elem_4, c_col1_4);
  b_elem_4 = _mm256_set1_pd(b_col0_4[2]);
  c_col_4 = _mm256_fmadd_pd(a_col20, b_elem_4, c_col_4);
  c_col1_4 = _mm256_fmadd_pd(a_col21, b_elem_4, c_col1_4);
  b_elem_4 = _mm256_set1_pd(b_col0_4[3]);
  c_col_4 = _mm256_fmadd_pd(a_col30, b_elem_4, c_col_4);
  c_col1_4 = _mm256_fmadd_pd(a_col31, b_elem_4, c_col1_4);

  // Repeat more with second half of B col
  b_elem_4 = _mm256_set1_pd(bcol1_4[0]);
  c_col_4 = _mm256_fmadd_pd(a_col40, b_elem_4, c_col_4);
  c_col1_4 = _mm256_fmadd_pd(a_col41, b_elem_4, c_col1_4);
  b_elem_4 = _mm256_set1_pd(bcol1_4[1]);
  c_col_4 = _mm256_fmadd_pd(a_col50, b_elem_4, c_col_4);
  c_col1_4 = _mm256_fmadd_pd(a_col51, b_elem_4, c_col1_4);
  b_elem_4 = _mm256_set1_pd(bcol1_4[2]);
  c_col_4 = _mm256_fmadd_pd(a_col60, b_elem_4, c_col_4);
  c_col1_4 = _mm256_fmadd_pd(a_col61, b_elem_4, c_col1_4);
  b_elem_4 = _mm256_set1_pd(bcol1_4[3]);
  c_col_4 = _mm256_fmadd_pd(a_col70, b_elem_4, c_col_4);
  c_col1_4 = _mm256_fmadd_pd(a_col71, b_elem_4, c_col1_4);

  // Store two halves of C col
  _mm256_storeu_pd(C + lda * 4, c_col_4);
  _mm256_storeu_pd(C + lda * 4 + 4, c_col1_4);

  //5

  __m256d b_col0_5 = _mm256_loadu_pd(B + lda * 5);
  __m256d bcol1_5 = _mm256_loadu_pd(B + lda * 5 + 4);
  // Load the C column in 2 pieces
  __m256d c_col_5 = _mm256_loadu_pd(C + lda * 5);
  __m256d c_col1_5 = _mm256_loadu_pd(C + lda * 5 + 4);

  // Broadcast the first element of the B col, dot with A col
  __m256d b_elem_5 = _mm256_set1_pd(b_col0_5[0]);
  c_col_5 = _mm256_fmadd_pd(a_col00, b_elem_5, c_col_5);
  // Repeat...
  c_col1_5 = _mm256_fmadd_pd(a_col01, b_elem_5, c_col1_5);
  b_elem_5 = _mm256_set1_pd(b_col0_5[1]);
  c_col_5 = _mm256_fmadd_pd(a_col10, b_elem_5, c_col_5);
  c_col1_5 = _mm256_fmadd_pd(a_col11, b_elem_5, c_col1_5);
  b_elem_5 = _mm256_set1_pd(b_col0_5[2]);
  c_col_5 = _mm256_fmadd_pd(a_col20, b_elem_5, c_col_5);
  c_col1_5 = _mm256_fmadd_pd(a_col21, b_elem_5, c_col1_5);
  b_elem_5 = _mm256_set1_pd(b_col0_5[3]);
  c_col_5 = _mm256_fmadd_pd(a_col30, b_elem_5, c_col_5);
  c_col1_5 = _mm256_fmadd_pd(a_col31, b_elem_5, c_col1_5);

  // Repeat more with second half of B col
  b_elem_5 = _mm256_set1_pd(bcol1_5[0]);
  c_col_5 = _mm256_fmadd_pd(a_col40, b_elem_5, c_col_5);
  c_col1_5 = _mm256_fmadd_pd(a_col41, b_elem_5, c_col1_5);
  b_elem_5 = _mm256_set1_pd(bcol1_5[1]);
  c_col_5 = _mm256_fmadd_pd(a_col50, b_elem_5, c_col_5);
  c_col1_5 = _mm256_fmadd_pd(a_col51, b_elem_5, c_col1_5);
  b_elem_5 = _mm256_set1_pd(bcol1_5[2]);
  c_col_5 = _mm256_fmadd_pd(a_col60, b_elem_5, c_col_5);
  c_col1_5 = _mm256_fmadd_pd(a_col61, b_elem_5, c_col1_5);
  b_elem_5 = _mm256_set1_pd(bcol1_5[3]);
  c_col_5 = _mm256_fmadd_pd(a_col70, b_elem_5, c_col_5);
  c_col1_5 = _mm256_fmadd_pd(a_col71, b_elem_5, c_col1_5);

  // Store two halves of C col
  _mm256_storeu_pd(C + lda * 5, c_col_5);
  _mm256_storeu_pd(C + lda * 5 + 4, c_col1_5);

  //6

  __m256d b_col0_6 = _mm256_loadu_pd(B + lda * 6);
  __m256d bcol1_6 = _mm256_loadu_pd(B + lda * 6 + 4);
  // Load the C column in 2 pieces
  __m256d c_col_6 = _mm256_loadu_pd(C + lda * 6);
  __m256d c_col1_6 = _mm256_loadu_pd(C + lda * 6 + 4);

  // Broadcast the first element of the B col, dot with A col
  __m256d b_elem_6 = _mm256_set1_pd(b_col0_6[0]);
  c_col_6 = _mm256_fmadd_pd(a_col00, b_elem_6, c_col_6);
  // Repeat...
  c_col1_6 = _mm256_fmadd_pd(a_col01, b_elem_6, c_col1_6);
  b_elem_6 = _mm256_set1_pd(b_col0_6[1]);
  c_col_6 = _mm256_fmadd_pd(a_col10, b_elem_6, c_col_6);
  c_col1_6 = _mm256_fmadd_pd(a_col11, b_elem_6, c_col1_6);
  b_elem_6 = _mm256_set1_pd(b_col0_6[2]);
  c_col_6 = _mm256_fmadd_pd(a_col20, b_elem_6, c_col_6);
  c_col1_6 = _mm256_fmadd_pd(a_col21, b_elem_6, c_col1_6);
  b_elem_6 = _mm256_set1_pd(b_col0_6[3]);
  c_col_6 = _mm256_fmadd_pd(a_col30, b_elem_6, c_col_6);
  c_col1_6 = _mm256_fmadd_pd(a_col31, b_elem_6, c_col1_6);

  // Repeat more with second half of B col
  b_elem_6 = _mm256_set1_pd(bcol1_6[0]);
  c_col_6 = _mm256_fmadd_pd(a_col40, b_elem_6, c_col_6);
  c_col1_6 = _mm256_fmadd_pd(a_col41, b_elem_6, c_col1_6);
  b_elem_6 = _mm256_set1_pd(bcol1_6[1]);
  c_col_6 = _mm256_fmadd_pd(a_col50, b_elem_6, c_col_6);
  c_col1_6 = _mm256_fmadd_pd(a_col51, b_elem_6, c_col1_6);
  b_elem_6 = _mm256_set1_pd(bcol1_6[2]);
  c_col_6 = _mm256_fmadd_pd(a_col60, b_elem_6, c_col_6);
  c_col1_6 = _mm256_fmadd_pd(a_col61, b_elem_6, c_col1_6);
  b_elem_6 = _mm256_set1_pd(bcol1_6[3]);
  c_col_6 = _mm256_fmadd_pd(a_col70, b_elem_6, c_col_6);
  c_col1_6 = _mm256_fmadd_pd(a_col71, b_elem_6, c_col1_6);

  // Store two halves of C col
  _mm256_storeu_pd(C + lda * 6, c_col_6);
  _mm256_storeu_pd(C + lda * 6 + 4, c_col1_6);

  //7

  __m256d b_col0_7 = _mm256_loadu_pd(B + lda * 7);
  __m256d bcol1_7 = _mm256_loadu_pd(B + lda * 7 + 4);
  // Load the C column in 2 pieces
  __m256d c_col_7 = _mm256_loadu_pd(C + lda * 7);
  __m256d c_col1_7 = _mm256_loadu_pd(C + lda * 7 + 4);

  // Broadcast the first element of the B col, dot with A col
  __m256d b_elem_7 = _mm256_set1_pd(b_col0_7[0]);
  c_col_7 = _mm256_fmadd_pd(a_col00, b_elem_7, c_col_7);
  // Repeat...
  c_col1_7 = _mm256_fmadd_pd(a_col01, b_elem_7, c_col1_7);
  b_elem_7 = _mm256_set1_pd(b_col0_7[1]);
  c_col_7 = _mm256_fmadd_pd(a_col10, b_elem_7, c_col_7);
  c_col1_7 = _mm256_fmadd_pd(a_col11, b_elem_7, c_col1_7);
  b_elem_7 = _mm256_set1_pd(b_col0_7[2]);
  c_col_7 = _mm256_fmadd_pd(a_col20, b_elem_7, c_col_7);
  c_col1_7 = _mm256_fmadd_pd(a_col21, b_elem_7, c_col1_7);
  b_elem_7 = _mm256_set1_pd(b_col0_7[3]);
  c_col_7 = _mm256_fmadd_pd(a_col30, b_elem_7, c_col_7);
  c_col1_7 = _mm256_fmadd_pd(a_col31, b_elem_7, c_col1_7);

  // Repeat more with second half of B col
  b_elem_7 = _mm256_set1_pd(bcol1_7[0]);
  c_col_7 = _mm256_fmadd_pd(a_col40, b_elem_7, c_col_7);
  c_col1_7 = _mm256_fmadd_pd(a_col41, b_elem_7, c_col1_7);
  b_elem_7 = _mm256_set1_pd(bcol1_7[1]);
  c_col_7 = _mm256_fmadd_pd(a_col50, b_elem_7, c_col_7);
  c_col1_7 = _mm256_fmadd_pd(a_col51, b_elem_7, c_col1_7);
  b_elem_7 = _mm256_set1_pd(bcol1_7[2]);
  c_col_7 = _mm256_fmadd_pd(a_col60, b_elem_7, c_col_7);
  c_col1_7 = _mm256_fmadd_pd(a_col61, b_elem_7, c_col1_7);
  b_elem_7 = _mm256_set1_pd(bcol1_7[3]);
  c_col_7 = _mm256_fmadd_pd(a_col70, b_elem_7, c_col_7);
  c_col1_7 = _mm256_fmadd_pd(a_col71, b_elem_7, c_col1_7);

  // Store two halves of C col
  _mm256_storeu_pd(C + lda * 7, c_col_7);
  _mm256_storeu_pd(C + lda * 7 + 4, c_col1_7);
  // for (int i = 0; i < 8; i++) {
  //   // Load the B column in 2 pieces
  //   __m256d b_col0 = _mm256_loadu_pd(B + lda * i);
  //   __m256d b_col1 = _mm256_loadu_pd(B + lda * i + 4);
  //   // Load the C column in 2 pieces
  //   __m256d c_col0 = _mm256_loadu_pd(C + lda * i);
  //   __m256d c_col1 = _mm256_loadu_pd(C + lda * i + 4);

  //   // Broadcast the first element of the B col, dot with A col
  //   __m256d b_elem = _mm256_set1_pd(b_col0[0]);
  //   c_col0 = _mm256_fmadd_pd(a_col00, b_elem, c_col0);
  //   // Repeat...
  //   c_col1 = _mm256_fmadd_pd(a_col01, b_elem, c_col1);
  //   b_elem = _mm256_set1_pd(b_col0[1]);
  //   c_col0 = _mm256_fmadd_pd(a_col10, b_elem, c_col0);
  //   c_col1 = _mm256_fmadd_pd(a_col11, b_elem, c_col1);
  //   b_elem = _mm256_set1_pd(b_col0[2]);
  //   c_col0 = _mm256_fmadd_pd(a_col20, b_elem, c_col0);
  //   c_col1 = _mm256_fmadd_pd(a_col21, b_elem, c_col1);
  //   b_elem = _mm256_set1_pd(b_col0[3]);
  //   c_col0 = _mm256_fmadd_pd(a_col30, b_elem, c_col0);
  //   c_col1 = _mm256_fmadd_pd(a_col31, b_elem, c_col1);

  //   // Repeat more with second half of B col
  //   b_elem = _mm256_set1_pd(b_col1[0]);
  //   c_col0 = _mm256_fmadd_pd(a_col40, b_elem, c_col0);
  //   c_col1 = _mm256_fmadd_pd(a_col41, b_elem, c_col1);
  //   b_elem = _mm256_set1_pd(b_col1[1]);
  //   c_col0 = _mm256_fmadd_pd(a_col50, b_elem, c_col0);
  //   c_col1 = _mm256_fmadd_pd(a_col51, b_elem, c_col1);
  //   b_elem = _mm256_set1_pd(b_col1[2]);
  //   c_col0 = _mm256_fmadd_pd(a_col60, b_elem, c_col0);
  //   c_col1 = _mm256_fmadd_pd(a_col61, b_elem, c_col1);
  //   b_elem = _mm256_set1_pd(b_col1[3]);
  //   c_col0 = _mm256_fmadd_pd(a_col70, b_elem, c_col0);
  //   c_col1 = _mm256_fmadd_pd(a_col71, b_elem, c_col1);

  //   // Store two halves of C col
  //   _mm256_storeu_pd(C + lda * i, c_col0);
  //   _mm256_storeu_pd(C + lda * i + 4, c_col1);
  // }
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
    double C[4][4];
    for (int i = 0; i < 4; i += 1)
        for (int j = 0; j < 4; j += 1)
            ((double*)C)[i+j*4] = ((double*)CT)[j+i*4];
    do_intrinsic(4, (double*)A, (double*)B, (double*)C);
//    do_reference(4, (double*)A, (double*)B, (double*)C);
    printf("---\n");
    printf("%f %f %f %f\n", C[0][0], C[1][0], C[2][0], C[3][0]);
    printf("%f %f %f %f\n", C[0][1], C[1][1], C[2][1], C[3][1]);
    printf("%f %f %f %f\n", C[0][2], C[1][2], C[2][2], C[3][2]);
    printf("%f %f %f %f\n", C[0][3], C[1][3], C[2][3], C[3][3]);
}

static void do_block_inner_ref (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j)
    {
      /* Compute C(i,j) */
      double cij = C[i+j*lda];
      // if (K == 4) {
      //   __m256d b = _mm256_loadu_pd(B + j*lda);
      //   __m256d a = _mm256_set_pd(A[i+0*lda], A[i+1*lda], A[i+2*lda], A[i+3*lda]);
      //   __m256d c_elem = _mm256_mul_pd(a, b);
      //   cij += c_elem[0] + c_elem[1] + c_elem[2] + c_elem[3];
      // } else {
      //   for (int k = 0; k < K; ++k)
      //     cij += A[i+k*lda] * B[k+j*lda];
      // }
      for (int k = 0; k < K; ++k)
        cij += A[i+k*lda] * B[k+j*lda];
      C[i+j*lda] = cij;
    }
}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block_inner (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  for (int j = 0; j < N; j += 8)
    for (int k = 0; k < K; k += 8)
      for (int i = 0; i < M; i += 8)
      {
        int M2 = min (8, M-i);
        int N2 = min (8, N-j);
        int K2 = min (8, K-k);
        if (M2 == 8 && N2 == 8 && K2 == 8) {
            do_intrinsic(lda, A + i + k*lda, B + k + j*lda, C + i + j*lda);
        } else {
            do_block_inner_ref(lda, M2, N2, K2, A + i + k*lda, B + k + j*lda, C + i + j*lda);
        }
      }
}

static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int j = 0; j < N; j += INNER_BLOCK_SIZE)
    /* For each column j of B */ 
    for (int k = 0; k < K; k += INNER_BLOCK_SIZE)
    {
      for (int i = 0; i < M; i += INNER_BLOCK_SIZE)
      {
          int M2 = min (INNER_BLOCK_SIZE, M-i);
          int N2 = min (INNER_BLOCK_SIZE, N-j);
          int K2 = min (INNER_BLOCK_SIZE, K-k);
          do_block_inner(lda, M2, N2, K2, A + i + k*lda, B + k + j*lda, C + i + j*lda);
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

  /* For each block-row of A */ 
  for (int j = 0; j < lda; j += BLOCK_SIZE)
    /* For each block-column of B */
    for (int k = 0; k < lda; k += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int i = 0; i < lda; i += BLOCK_SIZE)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M = min (BLOCK_SIZE, lda-i);
        int N = min (BLOCK_SIZE, lda-j);
        int K = min (BLOCK_SIZE, lda-k);

        /* Perform individual block dgemm */
        do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
}
