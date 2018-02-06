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

#include <stdio.h>
#include <immintrin.h>
//#include <iostream>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 256
#endif

#define INNER_BLOCK_SIZE 32
#define MIN_BLOCK_SIZE 4

#define min(a,b) (((a)<(b))?(a):(b))
//// HELPER FUNCTIONS ////

// Define a transpose function for a small matrix
void transpose(int lda, double *A, double *A_T) 
{
//  int j = 0;
  //int lim = (lda / 4) * 4;
  // Transpose block in sets of 4 x 4 (best for Cori)
  for (int i = 0; i < lda; i++) {
  /*  for (; j < lim; j += 4) {
      A_T[j + i * lda] = A[i + j * lda];
      A_T[(j+1) + i * lda] = A[i + (j+1) * lda]; 
      A_T[(j+2) + i * lda] = A[i + (j+2) * lda];
      A_T[(j+3) + i * lda] = A[i + (j+3) * lda];
    }*/
    // Handle the rest naively (e.g. edge bases in case lda is not divisibly by 4
    for (int j = 0; j < lda; j++) {
      A_T[j + i * lda] = A[i + j * lda];
    }
  }
}

//static void set_mat_mem(int lda, int M, int N, int K, double *A, double *B, double* Acurr, double* Bcurr)
//{
  //for (i = 0; i < (N - 3); i += 4) {

static void do_4x4(int lda, double* AT, double* B, double* C)
{
 __m256d A0x, A1x, A2x, A3x, Bx0, Bx1, Bx2, Bx3, C0x, C1x, C2x, C3x;

  //double* AT = malloc(sizeof(double)*lda*lda);
  //transpose(lda, A, AT);
  //lda = 4;
  int col1 = lda;
  int col2 = col1 + lda;
  int col3 = col2 + lda;

  A0x = _mm256_loadu_pd(AT);
  A1x = _mm256_loadu_pd(AT + col1);
  A2x = _mm256_loadu_pd(AT + col2);
  A3x = _mm256_loadu_pd(AT + col3);

  Bx0 = _mm256_loadu_pd(B);
  Bx1 = _mm256_loadu_pd(B + col1);
  Bx2 = _mm256_loadu_pd(B + col2);
  Bx3 = _mm256_loadu_pd(B + col3);
 
  C0x = _mm256_loadu_pd(C);
 // C0x1 = _mm256_loadu_pd(C + 1);
  //C0x2 = _mm256_loadu_pd(C + 2);
  //C0x3 = _mm256_loadu_pd(C + 3);
  C1x = _mm256_loadu_pd(C + col1);
  C2x = _mm256_loadu_pd(C + col2);
  C3x = _mm256_loadu_pd(C + col3);

  __m256d c = _mm256_setr_pd(C0x[0], 0.0, 0.0, 0.0);
  c = _mm256_fmadd_pd(A0x, Bx0, c);
  C0x[0] = c[0] + c[1] + c[2] + c[3];
  c = _mm256_setr_pd(C0x[1], 0.0, 0.0, 0.0);
  c = _mm256_fmadd_pd(A1x, Bx0, c);
  C0x[1] = c[0] + c[1] + c[2] + c[3];
  c = _mm256_setr_pd(C0x[2], 0.0, 0.0, 0.0);
  c = _mm256_fmadd_pd(A2x, Bx0, c);
  C0x[2] = c[0] + c[1] + c[2] + c[3];
  c = _mm256_setr_pd(C0x[3], 0.0, 0.0, 0.0);
  c = _mm256_fmadd_pd(A3x, Bx0, c);
  C0x[3] = c[0] + c[1] + c[2] + c[3];
  
  c = _mm256_setr_pd(C1x[0], 0.0, 0.0, 0.0);
  c = _mm256_fmadd_pd(A0x, Bx1, c);
  C1x[0] = c[0] + c[1] + c[2] + c[3];
  c = _mm256_setr_pd(C1x[1], 0.0, 0.0, 0.0);
  c = _mm256_fmadd_pd(A1x, Bx1, c);
  C1x[1] = c[0] + c[1] + c[2] + c[3];
  c = _mm256_setr_pd(C1x[2], 0.0, 0.0, 0.0);
  c = _mm256_fmadd_pd(A2x, Bx1, c);
  C1x[2] = c[0] + c[1] + c[2] + c[3];
  c = _mm256_setr_pd(C1x[3], 0.0, 0.0, 0.0);
  c = _mm256_fmadd_pd(A3x, Bx1, c);
  C1x[3] = c[0] + c[1] + c[2] + c[3];

  c = _mm256_setr_pd(C2x[0], 0.0, 0.0, 0.0);
  c = _mm256_fmadd_pd(A0x, Bx2, c);
  C2x[0] = c[0] + c[1] + c[2] + c[3];
  c = _mm256_setr_pd(C2x[1], 0.0, 0.0, 0.0);
  c = _mm256_fmadd_pd(A1x, Bx2, c);
  C2x[1] = c[0] + c[1] + c[2] + c[3];
  c = _mm256_setr_pd(C2x[2], 0.0, 0.0, 0.0);
  c = _mm256_fmadd_pd(A2x, Bx2, c);
  C2x[2] = c[0] + c[1] + c[2] + c[3];
  c = _mm256_setr_pd(C2x[3], 0.0, 0.0, 0.0);
  c = _mm256_fmadd_pd(A3x, Bx2, c);
  C2x[3] = c[0] + c[1] + c[2] + c[3];
  
  c = _mm256_setr_pd(C3x[0], 0.0, 0.0, 0.0);
  c = _mm256_fmadd_pd(A0x, Bx3, c);
  C3x[0] = c[0] + c[1] + c[2] + c[3];
  c = _mm256_setr_pd(C3x[1], 0.0, 0.0, 0.0);
  c = _mm256_fmadd_pd(A1x, Bx3, c);
  C3x[1] = c[0] + c[1] + c[2] + c[3];
  c = _mm256_setr_pd(C3x[2], 0.0, 0.0, 0.0);
  c = _mm256_fmadd_pd(A2x, Bx3, c);
  C3x[2] = c[0] + c[1] + c[2] + c[3];
  c = _mm256_setr_pd(C3x[3], 0.0, 0.0, 0.0);
  c = _mm256_fmadd_pd(A3x, Bx3, c);
  C3x[3] = c[0] + c[1] + c[2] + c[3];
  
  _mm256_storeu_pd(C, C0x);
  _mm256_storeu_pd(C + col1, C1x);
  _mm256_storeu_pd(C + col2, C2x);
  _mm256_storeu_pd(C + col3, C3x); 
}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block_inner (int lda, int M, int N, int K, double* AT, double* B, double* C)
{
  //double* AT = A;
  //transpose(lda, A, AT);
  
//  double* Acurrblock, Bcurrblock, Ccurrblock;
  

//  double* BT = B;
//  transpose(lda, B, BT);

  /* For each row i of A */
  for (int i = 0; i < M; ++i) {
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j) 
    {
      /* Compute C(i,j) */
      double cij = C[i + j*lda];
//      int k = 0;
/*      for (; k < K - 3; k+=4) {
        int Aoffset = k + i * lda;
        int Boffset = j + k * lda;
        double a = A[Aoffset] * B[Boffset];
        double b = A[Aoffset + lda] * B[Boffset + 1];
        double c = A[Aoffset + lda+lda] * B[Boffset + 2];
        double d = A[Aoffset + lda+lda+lda] * B[Boffset + 3];
//	__builtin_prefetch((const void *) &A[i+(k+8)*lda], 0, 3);
//	__builtin_prefetch((const void *) &B[(k+4)+j*lda], 0, 3);
	cij += a + b + c +d;
      } */
      for (int k = 0; k < K; ++k) {
        cij += AT[k + i*lda]*B[k + j*lda];
      }
      C[i + j*lda] = cij;
    }
    //printf("5");
  }
}
/*
static void test_4x4() 
{
  double A[4][4]= {
	{5, 6, 7, 8},
	{2, 2, 2, 2},
	{3, 3, 3, 3},
	{4, 4, 4, 4}};
  double B[4][4]= {
	{1, 10, 100, 1000},
	{1, 10, 100, 1000},
	{1, 10, 100, 1000},
	{1, 10, 100, 1000}};
  double C[4][4]= {
	{0, 0, 0, 0},
	{0, 0, 0, 0},
	{0, 0, 0, 0},
	{0, 0, 0, 0}};
  double C2[4][4]= {
	{0, 0, 0, 0},
	{0, 0, 0, 0},
	{0, 0, 0, 0},
	{0, 0, 0, 0}};
  
  do_4x4(4, (double *)A, (double *)B, (double *)C);
  //double C2[4][4];
  do_block_inner(4, 4, 4, 4, (double *)A, (double *)B, (double *)C2);
  //transpose(4, A, C2);  
  printf("-------------\n");
  printf("%f %f %f %f\n", A[0][0], A[1][0], A[2][0], A[3][0]);
  printf("%f %f %f %f\n", A[0][1], A[1][1], A[2][1], A[3][1]);
  printf("%f %f %f %f\n", A[0][2], A[1][2], A[2][2], A[3][2]);
  printf("%f %f %f %f\n", A[0][3], A[1][3], A[2][3], A[3][3]);
  

  printf("-------------\n");
  printf("%f %f %f %f\n", B[0][0], B[1][0], B[2][0], B[3][0]);
  printf("%f %f %f %f\n", B[0][1], B[1][1], B[2][1], B[3][1]);
  printf("%f %f %f %f\n", B[0][2], B[1][2], B[2][2], B[3][2]);
  printf("%f %f %f %f\n", B[0][3], B[1][3], B[2][3], B[3][3]);

  
  printf("-------------\n");
  printf("%f %f %f %f\n", C[0][0], C[1][0], C[2][0], C[3][0]);
  printf("%f %f %f %f\n", C[0][1], C[1][1], C[2][1], C[3][1]);
  printf("%f %f %f %f\n", C[0][2], C[1][2], C[2][2], C[3][2]);
  printf("%f %f %f %f\n", C[0][3], C[1][3], C[2][3], C[3][3]);
  
  printf("-------------\n");
  printf("%f %f %f %f\n", C2[0][0], C2[1][0], C2[2][0], C2[3][0]);
  printf("%f %f %f %f\n", C2[0][1], C2[1][1], C2[2][1], C2[3][1]);
  printf("%f %f %f %f\n", C2[0][2], C2[1][2], C2[2][2], C2[3][2]);
  printf("%f %f %f %f\n", C2[0][3], C2[1][3], C2[2][3], C2[3][3]);
}
*/
static void do_block_split(int lda, int M, int N, int K, double* AT, double* B, double* C) 
{
  // For each row i of A *
  for (int i = 0; i < M; i += MIN_BLOCK_SIZE) {
    // For each columb j of B *
    for (int j = 0; j < N; j += MIN_BLOCK_SIZE)
    {
      for (int k = 0; k < K ; k += MIN_BLOCK_SIZE){
         int M2 = min(MIN_BLOCK_SIZE, M-i);
         int N2 = min(MIN_BLOCK_SIZE, N-j);
         int K2 = min(MIN_BLOCK_SIZE, K-k);
         
         if (M2 == MIN_BLOCK_SIZE && N2 == MIN_BLOCK_SIZE && K2 == MIN_BLOCK_SIZE) {
           //double* A_temp = A;
           do_4x4(lda, AT + k + i*lda, B + k + j*lda, C + i + j*lda);
           //do_block_inner(lda, M2, N2, K2, A + i + k*lda, B + k + j*lda, C + i + j*lda);
           //if (A!= A_temp) printf("NO MATCH");
         }else {
           do_block_inner(lda, M2, N2, K2, AT + k + i*lda, B + k + j*lda, C + i + j*lda); 
         }   
      }
    }
  }
} 

static void do_block(int lda, int M, int N, int K, double* AT, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < M; i += INNER_BLOCK_SIZE) {
    /* For each columb j of B */
    for (int j = 0; j < N; j += INNER_BLOCK_SIZE)
    {
      for (int k = 0; k < K ; k += INNER_BLOCK_SIZE){
         int M2 = min(INNER_BLOCK_SIZE, M-i);
         int N2 = min(INNER_BLOCK_SIZE, N-j);
         int K2 = min(INNER_BLOCK_SIZE, K-k);
         
         do_block_split(lda, M2, N2, K2, AT + k + i*lda, B + k + j*lda, C + i + j*lda);   
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
  // let's transpose everything
  double* AT = malloc(sizeof(double)*lda*lda);
  //double* BT = malloc(sizeof(double)*lda*lda);
  //double* CT = malloc(sizeof(double)*lda*lda);
  transpose(lda, A, AT);
  //transpose(lda, B, BT);
  //transpose(lda, C, CT);
  //test_4x4();  
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
        //printf("10");
	/* Perform individual block dgemm */
	do_block(lda, M, N, K, AT + k + i*lda, B + k + j*lda, C + i + j*lda);
        //printf("11");
      }
    //printf("12");
    }
    //printf("13");
  }

  //transpose(lda, CT, C);
  //printf("14");
  free(AT);
  //free(BT);
  //free(CT);
}



