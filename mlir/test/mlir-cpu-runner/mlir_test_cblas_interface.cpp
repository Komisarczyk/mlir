//===- mlir_test_cblas_interface.cpp - Simple Blas subset interface -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simple Blas subset interface implementation.
//
//===----------------------------------------------------------------------===//

#include "include/mlir_test_cblas_interface.h"
//#include "include/mlir_test_cblas.h"
#include <assert.h>
#include <chrono>
#include <iostream>
#include <string.h>
#include <vector>
//#define HAS_CPU_SUPPORT
#define HAS_GPU_SUPPORT


//#include <dnnl.hpp>
//using namespace dnnl;

#ifdef HAS_GPU_SUPPORT
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

Timer *Timer::timer_instance = nullptr;

extern "C" void start_timer() { Timer::get_instance()->start_timer(); }

extern "C" void stop_timer() { Timer::get_instance()->stop_timer(); }

extern "C" void
_mlir_ciface_linalg_fill_viewf32_f32(StridedMemRefType<float, 0> *X, float f) {
  X->data[X->offset] = f;
}

extern "C" void
_mlir_ciface_linalg_fill_viewsxf32_f32(StridedMemRefType<float, 1> *X,
                                       float f) {
  for (unsigned i = 0; i < X->sizes[0]; ++i) {
    *(X->data + X->offset + i * X->strides[0]) = f;
    f++;
  }
}

extern "C" void
_mlir_ciface_linalg_fill_viewsxsxf32_f32(StridedMemRefType<float, 2> *X,
                                         float f) {
  for (unsigned i = 0; i < X->sizes[0]; ++i)
    for (unsigned j = 0; j < X->sizes[1]; ++j) {
      *(X->data + X->offset + i * X->strides[0] + j * X->strides[1]) = f;
      f++;
    }
}
void printmatrix(float *A, int M, int N) {

  for (int64_t x = 0; x < M; x++) {
    for (int64_t y = 0; y < N; y++) {
      printf("%f%s", *(A + x * N + y), y == N - 1 ? "" : " ");
    }

    puts("");
  }
}
extern "C" void
_mlir_ciface_linalg_fill_viewsxsxsxf32_f32_f32(StridedMemRefType<float, 3> *X,
                                               float f) {
  for (unsigned i = 0; i < X->sizes[0]; ++i)
    for (unsigned j = 0; j < X->sizes[1]; ++j)
      for (unsigned k = 0; k < X->sizes[2]; ++k) {
        *(X->data + X->offset + i * X->strides[0] + j * X->strides[1] +
          k * X->strides[2]) = f;
        f++;
      }
}

extern "C" void _mlir_ciface_linalg_fill_viewsxsxsxsxf32_f32_f32_f32(
    StridedMemRefType<float, 4> *X, float f) {
  for (unsigned i = 0; i < X->sizes[0]; ++i)
    for (unsigned j = 0; j < X->sizes[1]; ++j)
      for (unsigned k = 0; k < X->sizes[2]; ++k)
        for (unsigned l = 0; l < X->sizes[3]; l++) {
          *(X->data + X->offset + i * X->strides[0] + j * X->strides[1] +
            k * X->strides[2] + l * X->strides[3]) = f;
          f++;
        }
}

extern "C" void
_mlir_ciface_linalg_copy_viewf32_viewf32(StridedMemRefType<float, 0> *I,
                                         StridedMemRefType<float, 0> *O) {
  O->data[O->offset] = I->data[I->offset];
}

extern "C" void
_mlir_ciface_linalg_copy_viewsxf32_viewsxf32(StridedMemRefType<float, 1> *I,
                                             StridedMemRefType<float, 1> *O) {
  if (I->sizes[0] != O->sizes[0]) {
    std::cerr << "Incompatible strided memrefs\n";
    printMemRefMetaData(std::cerr, *I);
    printMemRefMetaData(std::cerr, *O);
    return;
  }
  for (unsigned i = 0; i < I->sizes[0]; ++i)
    O->data[O->offset + i * O->strides[0]] =
        I->data[I->offset + i * I->strides[0]];
}

extern "C" void _mlir_ciface_linalg_copy_viewsxsxf32_viewsxsxf32(
    StridedMemRefType<float, 2> *I, StridedMemRefType<float, 2> *O) {
  if (I->sizes[0] != O->sizes[0] || I->sizes[1] != O->sizes[1]) {
    std::cerr << "Incompatible strided memrefs\n";
    printMemRefMetaData(std::cerr, *I);
    printMemRefMetaData(std::cerr, *O);
    return;
  }
  auto so0 = O->strides[0], so1 = O->strides[1];
  auto si0 = I->strides[0], si1 = I->strides[1];
  for (unsigned i = 0; i < I->sizes[0]; ++i)
    for (unsigned j = 0; j < I->sizes[1]; ++j)
      O->data[O->offset + i * so0 + j * so1] =
          I->data[I->offset + i * si0 + j * si1];
}
/*
extern "C" void _mlir_ciface_linalg_dot_viewsxf32_viewsxf32_viewf32(
    StridedMemRefType<float, 1> *X, StridedMemRefType<float, 1> *Y,
    StridedMemRefType<float, 0> *Z) {
  if (X->strides[0] != 1 || Y->strides[0] != 1 || X->sizes[0] != Y->sizes[0]) {
    std::cerr << "Incompatible strided memrefs\n";
    printMemRefMetaData(std::cerr, *X);
    printMemRefMetaData(std::cerr, *Y);
    printMemRefMetaData(std::cerr, *Z);
    return;
  }
  Z->data[Z->offset] +=
      mlir_test_cblas_sdot(X->sizes[0], X->data + X->offset, X->strides[0],
                           Y->data + Y->offset, Y->strides[0]);
}*/

static inline int compare(float *a, float *b, int M, int N) {

  /* Compare elements */
  for (int64_t x = 0; x < M; x++)
    for (int64_t y = 0; y < N; y++)

      if (a[x * N + y] != b[x * N + y]) {
        printf("\n %f %f  %d %d\n", a[x * N + y], b[x * N + y], x, y);
        return 0;
      }

  return 1;
}
/*
void matmulBlas(int transA, int transB, StridedMemRefType<float, 2> &C,
                StridedMemRefType<float, 2> &A, StridedMemRefType<float, 2> &B,
                int alpha, int beta) {
  size_t M = C.sizes[0];
  size_t N = C.sizes[1];
  size_t K = A.sizes[1];
  size_t lda = K;
  size_t ldb = N;
  size_t ldc = N;
  char isTransA = (transA) ? 'T' : 'N';
  char isTransB = (transB) ? 'T' : 'N';
#ifdef HAS_CPU_SUPPORT
  auto res =
      sgemm(isTransA, isTransB, M, N, K, (float)alpha, A.data + A.offset, lda,
            B.data + B.offset, ldb, (float)beta, C.data + C.offset, ldc);
  if (res != status::success)
    assert(0 && "sgemm failed");

  return;
#endif

  assert(0 && "naive gemm not implemented yet");
}*/
extern "C" void blasMatmul( float *B_allocatedptr, float *B_alignedptr,
                           int64_t B_offset, int64_t B_sizes0, int64_t B_sizes1,
                           int64_t B_strides0, int64_t B_strides1,
                           float *A_allocatedptr, float *A_alignedptr,
                           int64_t A_offset, int64_t A_sizes0, int64_t A_sizes1,
                           int64_t A_strides0, int64_t A_strides1,
                           float *C_allocatedptr, float *C_alignedptr,
                           int64_t C_offset, int64_t C_sizes0, int64_t C_sizes1,
                           int64_t C_strides0, int64_t C_strides1
                          ) {

  size_t M = C_sizes0;
  size_t N = C_sizes1;
  size_t K = A_sizes1;
  /*
  StridedMemRefType<float, 2> C;
  C.basePtr = C_allocatedptr;
  C.data = C_alignedptr;
  C.offset = C_offset;
  C.sizes[0] = C_sizes0;
  C.sizes[1] = C_sizes1;
  C.strides[0] = C_strides0;
  C.strides[1] = C_strides1;
  StridedMemRefType<float, 2> A;
  A.basePtr = A_allocatedptr;
  A.data = A_alignedptr;
  A.offset = A_offset;
  A.sizes[0] = A_sizes0;
  A.sizes[1] = A_sizes1;
  A.strides[0] = A_strides0;
  A.strides[1] = A_strides1;
  StridedMemRefType<float, 2> B;
  B.basePtr = B_allocatedptr;
  B.data = B_alignedptr;
  B.offset = B_offset;
  B.sizes[0] = B_sizes0;
  B.sizes[1] = B_sizes1;
  B.strides[0] = B_strides0;
  B.strides[1] = B_strides1;
  
  // no need for dimForM, N and K as the memref is 2d.
  if (A_strides1 != B_strides1 || A_strides1 != C_strides1 || A_strides1 != 1 ||
      A_sizes0 < A_strides1 || B_sizes0 < B_strides1 || C_sizes0 < C_strides1 ||
      C_sizes0 != A_sizes0 || C_sizes1 != B_sizes1 || A_sizes1 != B_sizes0) {
    printMemRefMetaData(std::cerr, A);
    printMemRefMetaData(std::cerr, B);
    printMemRefMetaData(std::cerr, C);
    return;
  }
  */
/*
  cblas_sgemm('N', 'N', C_sizes0, C_sizes1, A_sizes1, 1.0f,
        A_alignedptr + A_offset, A_sizes1, B_alignedptr + B_offset, C_sizes1,
        2.0f, C_alignedptr + C_offset, C_sizes1);*/

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C_sizes0, C_sizes1, A_sizes1, 1.0f,
        A_alignedptr + A_offset, A_sizes1, B_alignedptr + B_offset, C_sizes1,
        1.0f, C_alignedptr + C_offset, C_sizes1);
}

extern "C" void blasMatmul1( float *B_allocatedptr, float *B_alignedptr,
                           int64_t B_offset, int64_t B_sizes0, int64_t B_sizes1,
                           int64_t B_strides0, int64_t B_strides1,
                           float *A_allocatedptr, float *A_alignedptr,
                           int64_t A_offset, int64_t A_sizes0, int64_t A_sizes1,
                           int64_t A_strides0, int64_t A_strides1,
                           float *C_allocatedptr, float *C_alignedptr,
                           int64_t C_offset, int64_t C_sizes0, int64_t C_sizes1,
                           int64_t C_strides0, int64_t C_strides1
                          ) {

  size_t M = C_sizes0;
  size_t N = C_sizes1;
  size_t K = A_sizes1;

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C_sizes0, C_sizes1, A_sizes1, 1.0f,
        A_alignedptr + A_offset, A_sizes1, B_alignedptr + B_offset, C_sizes1,
        1.0f, C_alignedptr + C_offset, C_sizes1);
}

extern "C" void blasMatmul2( float *B_allocatedptr, float *B_alignedptr,
                           int64_t B_offset, int64_t B_sizes0, int64_t B_sizes1,
                           int64_t B_strides0, int64_t B_strides1,
                           float *A_allocatedptr, float *A_alignedptr,
                           int64_t A_offset, int64_t A_sizes0, int64_t A_sizes1,
                           int64_t A_strides0, int64_t A_strides1,
                           float *C_allocatedptr, float *C_alignedptr,
                           int64_t C_offset, int64_t C_sizes0, int64_t C_sizes1,
                           int64_t C_strides0, int64_t C_strides1
                          ) {

  size_t M = C_sizes0;
  size_t N = C_sizes1;
  size_t K = A_sizes1;

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C_sizes0, C_sizes1, A_sizes1, 1.0f,
        A_alignedptr + A_offset, A_sizes1, B_alignedptr + B_offset, C_sizes1,
        1.0f, C_alignedptr + C_offset, C_sizes1);
}
extern "C" void blasMatmulTime( float *B_allocatedptr, float *B_alignedptr,
                           int64_t B_offset, int64_t B_sizes0, int64_t B_sizes1,
                           int64_t B_strides0, int64_t B_strides1,
                           float *A_allocatedptr, float *A_alignedptr,
                           int64_t A_offset, int64_t A_sizes0, int64_t A_sizes1,
                           int64_t A_strides0, int64_t A_strides1,
                           float *C_allocatedptr, float *C_alignedptr,
                           int64_t C_offset, int64_t C_sizes0, int64_t C_sizes1,
                           int64_t C_strides0, int64_t C_strides1
                          ) {

         int LOOP_COUNT = 100000;int r;
  auto s_initial = dsecnd();
    for ( r = 0; r < LOOP_COUNT; r++) {

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C_sizes0, C_sizes1, A_sizes1, 1.0f,
        A_alignedptr + A_offset, A_sizes1, B_alignedptr + B_offset, C_sizes1,
        1.0f, C_alignedptr + C_offset, C_sizes1);
    }
   auto  s_elapsed = (dsecnd() - s_initial) / 1;
   printf ("+%d", ((int)(s_elapsed * 1000)));
}
extern "C" void blasMatmulTime1( float *B_allocatedptr, float *B_alignedptr,
                           int64_t B_offset, int64_t B_sizes0, int64_t B_sizes1,
                           int64_t B_strides0, int64_t B_strides1,
                           float *A_allocatedptr, float *A_alignedptr,
                           int64_t A_offset, int64_t A_sizes0, int64_t A_sizes1,
                           int64_t A_strides0, int64_t A_strides1,
                           float *C_allocatedptr, float *C_alignedptr,
                           int64_t C_offset, int64_t C_sizes0, int64_t C_sizes1,
                           int64_t C_strides0, int64_t C_strides1
                          ) {

         int LOOP_COUNT = 100000;int r;
  auto s_initial = dsecnd();
    for ( r = 0; r < LOOP_COUNT; r++) {

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C_sizes0, C_sizes1, A_sizes1, 1.0f,
        A_alignedptr + A_offset, A_sizes1, B_alignedptr + B_offset, C_sizes1,
        1.0f, C_alignedptr + C_offset, C_sizes1);
    }
   auto  s_elapsed = (dsecnd() - s_initial) / 1;
   printf ("+%d", ((int)(s_elapsed * 1000)));
}
extern "C" void blasMatmulTime2( float *B_allocatedptr, float *B_alignedptr,
                           int64_t B_offset, int64_t B_sizes0, int64_t B_sizes1,
                           int64_t B_strides0, int64_t B_strides1,
                           float *A_allocatedptr, float *A_alignedptr,
                           int64_t A_offset, int64_t A_sizes0, int64_t A_sizes1,
                           int64_t A_strides0, int64_t A_strides1,
                           float *C_allocatedptr, float *C_alignedptr,
                           int64_t C_offset, int64_t C_sizes0, int64_t C_sizes1,
                           int64_t C_strides0, int64_t C_strides1
                          ) {

         int LOOP_COUNT = 100000;int r;
  auto s_initial = dsecnd();
    for ( r = 0; r < LOOP_COUNT; r++) {

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C_sizes0, C_sizes1, A_sizes1, 1.0f,
        A_alignedptr + A_offset, A_sizes1, B_alignedptr + B_offset, C_sizes1,
        1.0f, C_alignedptr + C_offset, C_sizes1);
    }
   auto  s_elapsed = (dsecnd() - s_initial) / 1;
   printf ("+%d", ((int)(s_elapsed * 1000)));
}

extern "C" void blasMatvec(
    float *C_allocatedptr, float *C_alignedptr,
    int64_t C_offset, int64_t C_sizes0, int64_t C_sizes1, int64_t C_strides0,
    int64_t C_strides1, float *A_allocatedptr, float *A_alignedptr,
    int64_t A_offset, int64_t A_sizes0, int64_t A_strides0,
    float *B_allocatedptr, float *B_alignedptr,
    int64_t B_offset, int64_t B_sizes0, int64_t B_strides0) {


  cblas_sgemv(CblasRowMajor, CblasNoTrans, C_sizes0, C_sizes1, 1.0f,
  C_alignedptr + C_offset, C_sizes1,
  A_alignedptr + A_offset, 1, 1.0f,
  B_alignedptr + B_offset, 1);

}
extern "C" void blasMatvecTime(
    float *C_allocatedptr, float *C_alignedptr,
    int64_t C_offset, int64_t C_sizes0, int64_t C_sizes1, int64_t C_strides0,
    int64_t C_strides1, float *A_allocatedptr, float *A_alignedptr,
    int64_t A_offset, int64_t A_sizes0, int64_t A_strides0,
    float *B_allocatedptr, float *B_alignedptr,
    int64_t B_offset, int64_t B_sizes0, int64_t B_strides0) {
mkl_verbose(0);
mkl_set_dynamic( 0);
mkl_set_num_threads(8);
  size_t M = C_sizes0;
  size_t N = C_sizes1;
 int LOOP_COUNT = 100000;int r;
  auto s_initial = dsecnd();
    for ( r = 0; r < LOOP_COUNT; r++) {
  cblas_sgemv(CblasRowMajor, CblasNoTrans, C_sizes0, C_sizes1, 1.0f,
  C_alignedptr + C_offset, C_sizes1,
  A_alignedptr + A_offset, 1, 1.0f,
  B_alignedptr + B_offset, 1);
    }
   auto  s_elapsed = (dsecnd() - s_initial) / 1;
   printf (" +%f ", ((s_elapsed * 1000)));
}
extern "C" void blasVecmatTime(
    float *C_allocatedptr, float *C_alignedptr,
    int64_t C_offset, int64_t C_sizes0, int64_t C_sizes1, int64_t C_strides0,
    int64_t C_strides1, float *A_allocatedptr, float *A_alignedptr,
    int64_t A_offset, int64_t A_sizes0, int64_t A_strides0,
    float *B_allocatedptr, float *B_alignedptr,
    int64_t B_offset, int64_t B_sizes0, int64_t B_strides0) {

  size_t M = C_sizes0;
  size_t N = C_sizes1;
mkl_verbose(0);
mkl_set_dynamic( 0);
mkl_set_num_threads(8);
  int LOOP_COUNT = 100000;int r;
  auto s_initial = dsecnd();
    for ( r = 0; r < LOOP_COUNT; r++) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        1, N, M,
        1.0f,
        A_alignedptr + A_offset, M, C_alignedptr + C_offset, N,
        1.0f, B_alignedptr + B_offset, N);
    }
   auto  s_elapsed = (dsecnd() - s_initial) / 1;
   printf (" +%f ", ((s_elapsed * 1000)));

}

extern "C" void blasVecmat(
    float *C_allocatedptr, float *C_alignedptr,
    int64_t C_offset, int64_t C_sizes0, int64_t C_sizes1, int64_t C_strides0,
    int64_t C_strides1, float *A_allocatedptr, float *A_alignedptr,
    int64_t A_offset, int64_t A_sizes0, int64_t A_strides0,
    float *B_allocatedptr, float *B_alignedptr,
    int64_t B_offset, int64_t B_sizes0, int64_t B_strides0) {

  size_t M = C_sizes0;
  size_t N = C_sizes1;

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        1, N, M,
        1.0f,
        A_alignedptr + A_offset, M, C_alignedptr + C_offset, N,
        1.0f, B_alignedptr + B_offset, N);
    


}

extern "C" void init_matrix(float *B_allocatedptr,
             float *B_alignedptr, int64_t B_offset, int64_t B_sizes0,
             int64_t B_sizes1, int64_t B_strides0, int64_t B_strides1) {
      for (int64_t x = 0; x < B_sizes0; x++)
  for (int64_t y = 0; y < B_sizes1; y++)
   
      *(B_alignedptr + y + x*B_sizes1) = (float)(x - y);
}

/* Initialize vector with value x at position (x) */
extern "C" void init_vector(float *A_allocatedptr, float *A_alignedptr,
             int64_t A_offset, int64_t A_sizes0, int64_t A_strides0) {
  for (int64_t x = 0; x < A_sizes0; x++)
    *(A_alignedptr + x) = (float)(x);
}
extern "C" void
cublasMatmul(
  float *B_allocatedptr,
             float *B_alignedptr, int64_t B_offset, int64_t B_sizes0,
             int64_t B_sizes1, int64_t B_strides0, int64_t B_strides1,

             float *A_allocatedptr, float *A_alignedptr,
             int64_t A_offset, int64_t A_sizes0, int64_t A_sizes1,
             int64_t A_strides0, int64_t A_strides1,


            float *C_allocatedptr, float *C_alignedptr, int64_t C_offset,
             int64_t C_sizes0, int64_t C_sizes1, int64_t C_strides0,
             int64_t C_strides1 ) {
#ifdef HAS_GPU_SUPPORT

  size_t M = C_sizes0;
  size_t N = C_sizes1;
  size_t K = A_sizes1;

  matmulcuBlas(C_alignedptr, A_alignedptr, B_alignedptr, M, N, K);
#endif
}
extern "C" void
cublasMatmulTime(
  float *B_allocatedptr,
             float *B_alignedptr, int64_t B_offset, int64_t B_sizes0,
             int64_t B_sizes1, int64_t B_strides0, int64_t B_strides1,

             float *A_allocatedptr, float *A_alignedptr,
             int64_t A_offset, int64_t A_sizes0, int64_t A_sizes1,
             int64_t A_strides0, int64_t A_strides1,


            float *C_allocatedptr, float *C_alignedptr, int64_t C_offset,
             int64_t C_sizes0, int64_t C_sizes1, int64_t C_strides0,
             int64_t C_strides1 ) {
#ifdef HAS_GPU_SUPPORT

  size_t M = C_sizes0;
  size_t N = C_sizes1;
  size_t K = A_sizes1;
  int LOOP_COUNT = 100000;int r;
  auto s_initial = dsecnd();
    for ( r = 0; r < LOOP_COUNT; r++) {
  matmulcuBlas(C_alignedptr, A_alignedptr, B_alignedptr, M, N, K);
    }
   auto  s_elapsed = (dsecnd() - s_initial) / 1;
     printf (" +%f ", ((s_elapsed)));

#endif
}

extern "C" void
cublasMatmul1(
  float *B_allocatedptr,
             float *B_alignedptr, int64_t B_offset, int64_t B_sizes0,
             int64_t B_sizes1, int64_t B_strides0, int64_t B_strides1,

             float *A_allocatedptr, float *A_alignedptr,
             int64_t A_offset, int64_t A_sizes0, int64_t A_sizes1,
             int64_t A_strides0, int64_t A_strides1,


            float *C_allocatedptr, float *C_alignedptr, int64_t C_offset,
             int64_t C_sizes0, int64_t C_sizes1, int64_t C_strides0,
             int64_t C_strides1 ){

  size_t M = C_sizes0;
  size_t N = C_sizes1;
  size_t K = A_sizes1;

  matmulcuBlas(C_alignedptr, A_alignedptr, B_alignedptr, M, N, K);

}

extern "C" void
cublasMatmulTime1(
  float *B_allocatedptr,
             float *B_alignedptr, int64_t B_offset, int64_t B_sizes0,
             int64_t B_sizes1, int64_t B_strides0, int64_t B_strides1,

             float *A_allocatedptr, float *A_alignedptr,
             int64_t A_offset, int64_t A_sizes0, int64_t A_sizes1,
             int64_t A_strides0, int64_t A_strides1,


            float *C_allocatedptr, float *C_alignedptr, int64_t C_offset,
             int64_t C_sizes0, int64_t C_sizes1, int64_t C_strides0,
             int64_t C_strides1 ) {

  size_t M = C_sizes0;
  size_t N = C_sizes1;
  size_t K = A_sizes1;
  int LOOP_COUNT = 100000;int r;
  auto s_initial = dsecnd();
    for ( r = 0; r < LOOP_COUNT; r++) {
matmulcuBlas(C_alignedptr, A_alignedptr, B_alignedptr, M, N, K);
    }
   auto  s_elapsed = (dsecnd() - s_initial) / 1;
     printf (" +%f ", ((s_elapsed)));

}
extern "C" void
cublasMatmul2(
  float *B_allocatedptr,
             float *B_alignedptr, int64_t B_offset, int64_t B_sizes0,
             int64_t B_sizes1, int64_t B_strides0, int64_t B_strides1,

             float *A_allocatedptr, float *A_alignedptr,
             int64_t A_offset, int64_t A_sizes0, int64_t A_sizes1,
             int64_t A_strides0, int64_t A_strides1,


            float *C_allocatedptr, float *C_alignedptr, int64_t C_offset,
             int64_t C_sizes0, int64_t C_sizes1, int64_t C_strides0,
             int64_t C_strides1 ){

  size_t M = C_sizes0;
  size_t N = C_sizes1;
  size_t K = A_sizes1;

  matmulcuBlas(C_alignedptr, A_alignedptr, B_alignedptr, M, N, K);

}
extern "C" void
cublasMatmulTime2(
  float *B_allocatedptr,
             float *B_alignedptr, int64_t B_offset, int64_t B_sizes0,
             int64_t B_sizes1, int64_t B_strides0, int64_t B_strides1,

             float *A_allocatedptr, float *A_alignedptr,
             int64_t A_offset, int64_t A_sizes0, int64_t A_sizes1,
             int64_t A_strides0, int64_t A_strides1,


            float *C_allocatedptr, float *C_alignedptr, int64_t C_offset,
             int64_t C_sizes0, int64_t C_sizes1, int64_t C_strides0,
             int64_t C_strides1 ) {

  size_t M = C_sizes0;
  size_t N = C_sizes1;
  size_t K = A_sizes1;
  int LOOP_COUNT = 100000;int r;
  auto s_initial = dsecnd();
    for ( r = 0; r < LOOP_COUNT; r++) {
matmulcuBlas(C_alignedptr, A_alignedptr, B_alignedptr, M, N, K);
    }
   auto  s_elapsed = (dsecnd() - s_initial) / 1;
     printf (" +%f ", ((s_elapsed)));

}
extern "C" void
cublasMatvec(float *C_allocatedptr, float *C_alignedptr, int64_t C_offset,
             int64_t C_sizes0, int64_t C_sizes1, int64_t C_strides0,
             int64_t C_strides1, float *A_allocatedptr, float *A_alignedptr,
             int64_t A_offset, int64_t A_sizes0, int64_t A_strides0,
             float *B_allocatedptr, float *B_alignedptr, int64_t B_offset,
             int64_t B_sizes0, int64_t B_strides0) {

#ifdef HAS_GPU_SUPPORT

  size_t M = C_sizes0;
  size_t N = C_sizes1;
  size_t K = 1;
  matveccuBlas(C_alignedptr, A_alignedptr, B_alignedptr, M, N, K);
#endif
}
extern "C" void
cublasMatvecTime(float *C_allocatedptr, float *C_alignedptr, int64_t C_offset,
             int64_t C_sizes0, int64_t C_sizes1, int64_t C_strides0,
             int64_t C_strides1, float *A_allocatedptr, float *A_alignedptr,
             int64_t A_offset, int64_t A_sizes0, int64_t A_strides0,
             float *B_allocatedptr, float *B_alignedptr, int64_t B_offset,
             int64_t B_sizes0, int64_t B_strides0) {

#ifdef HAS_GPU_SUPPORT

  size_t M = C_sizes0;
  size_t N = C_sizes1;
  size_t K = 1;


    int LOOP_COUNT = 100000;int r;
  auto s_initial = dsecnd();
    for ( r = 0; r < LOOP_COUNT; r++) {
  matveccuBlas(C_alignedptr, A_alignedptr, B_alignedptr, M, N, K);
    }
   auto  s_elapsed = (dsecnd() - s_initial) / 1;
     printf (" +%f ", ((s_elapsed)));
#endif
}
extern "C" void
cublasVecmat(float *C_allocatedptr, float *C_alignedptr, int64_t C_offset,
             int64_t C_sizes0, int64_t C_sizes1, int64_t C_strides0,
             int64_t C_strides1, float *A_allocatedptr, float *A_alignedptr,
             int64_t A_offset, int64_t A_sizes0, int64_t A_strides0,
             float *B_allocatedptr, float *B_alignedptr, int64_t B_offset,
             int64_t B_sizes0, int64_t B_strides0) {

#ifdef HAS_GPU_SUPPORT

  size_t M = C_sizes0;
  size_t N = C_sizes1;
  size_t K = 1;

  vecmatcuBlas(C_alignedptr, A_alignedptr, B_alignedptr, M, N, K);
#endif
}
extern "C" void
cublasVecmatTime(float *C_allocatedptr, float *C_alignedptr, int64_t C_offset,
             int64_t C_sizes0, int64_t C_sizes1, int64_t C_strides0,
             int64_t C_strides1, float *A_allocatedptr, float *A_alignedptr,
             int64_t A_offset, int64_t A_sizes0, int64_t A_strides0,
             float *B_allocatedptr, float *B_alignedptr, int64_t B_offset,
             int64_t B_sizes0, int64_t B_strides0) {

#ifdef HAS_GPU_SUPPORT

  size_t M = C_sizes0;
  size_t N = C_sizes1;
  size_t K = 1;
    int LOOP_COUNT = 100000;int r;
  auto s_initial = dsecnd();
    for ( r = 0; r < LOOP_COUNT; r++) {
  vecmatcuBlas(C_alignedptr, A_alignedptr, B_alignedptr, M, N, K);
    }
   auto  s_elapsed = (dsecnd() - s_initial) / 1;
     printf (" +%f ", ((s_elapsed)));

#endif
}
extern "C" void
_mlir_ciface_linalg_fill_view42x42xf32_f32(StridedMemRefType<float, 2> *X,
                                           float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view3x5x4xf32_f32(StridedMemRefType<float, 3> *X,
                                           float f) {
  _mlir_ciface_linalg_fill_viewsxsxsxf32_f32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view2x3x4xf32_f32(StridedMemRefType<float, 3> *X,
                                           float f) {
  _mlir_ciface_linalg_fill_viewsxsxsxf32_f32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view2x5xf32_f32(StridedMemRefType<float, 2> *X,
                                         float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view1200x1200xf32_f32(StridedMemRefType<float, 2> *X,
                                               float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view1200x1000xf32_f32(StridedMemRefType<float, 2> *X,
                                               float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

#ifdef HAS_TRANSPOSE_SUPPORT
template <int D>
inline memory::dims shapeToMklDnnDims(const StridedMemRefType<float, D> *T) {
  memory::dims dims(D);
  for (int d = 0; d < D; ++d) {
    dims[d] = T->sizes[d];
  }
  return dims;
}

inline memory::dims calculateStrides(const memory::dims &dimsOrder) {
  memory::dims strides(dimsOrder.size());
  int lastDimIdx = dimsOrder.size() - 1;
  strides[lastDimIdx] = 1;
  for (int d = lastDimIdx - 1; d >= 0; d--) {
    strides[d] = strides[d + 1] * dimsOrder[d + 1];
  }
  return strides;
}

inline memory::dims reorderStrides(const memory::dims &strides,
                                   std::vector<int> perm) {
  memory::dims reordered_strides;
  reordered_strides.resize(strides.size());
  for (size_t i = 0; i < strides.size(); ++i) {
    reordered_strides[perm[i]] = strides[i];
  }
  return reordered_strides;
}

void createBlockedMemDescHelper(const memory::dims &dims,
                                const memory::dims &strides,
                                dnnl_memory_desc_t *blocked_md) {
  const int k_num_dims = dims.size();
  dnnl_dim_t input_dims[k_num_dims];
  dnnl_dim_t input_strides[k_num_dims];
  for (int i = 0; i < k_num_dims; ++i) {
    input_dims[i] = dims[i];
    input_strides[i] = strides[i];
  }
  dnnl_memory_desc_init_by_strides(blocked_md, k_num_dims, input_dims, dnnl_f32,
                                   input_strides);
}

inline memory::desc getMemDescr(const memory::dims &dims,
                                const memory::dims &strides) {
  dnnl_memory_desc_t blocked_md;
  createBlockedMemDescHelper(dims, strides, &blocked_md);
  return memory::desc(blocked_md);
}
#endif

template <int T, int Z>
void transposeBlas(StridedMemRefType<float, T> *S,
                   StridedMemRefType<float, Z> *D, int *perm, int s) {
  // std::cout << "\nSource -> \n";
  // printMemRefMetaData(std::cerr, *S);
  // std::cout << "\nDest -> \n";
  // printMemRefMetaData(std::cerr, *D);
  // std::cout << "\n\n";

  std::vector<int> arrayPerm{};
  for (int i = 0; i < s; i++)
    arrayPerm.push_back(*(perm++));

    // std::cout << "\nPermutation -> \n";
    // for (const auto elem : arrayPerm)
    //  std::cout << elem << "\n";

#ifdef HAS_TRANSPOSE_SUPPORT
  auto cpu_engine = engine(engine::kind::cpu, 0);
  memory::dims in_dims = shapeToMklDnnDims(S);
  memory::dims out_dims = shapeToMklDnnDims(D);
  memory::dims in_strides = calculateStrides(in_dims);
  memory::dims out_strides =
      reorderStrides(calculateStrides(out_dims), arrayPerm);
  auto inputMemDescr = getMemDescr(in_dims, in_strides);
  auto outputMemDescr = getMemDescr(in_dims, out_strides);
  auto inputMemory = memory(inputMemDescr, cpu_engine, S->data + S->offset);
  auto outputMemory = memory(outputMemDescr, cpu_engine, D->data + D->offset);
  auto r1 = reorder(inputMemory, outputMemory);
  auto stream_cpu = stream(cpu_engine);
  r1.execute(stream_cpu, inputMemory, outputMemory);
  return;
#endif

  assert(0 && "naive transpose not implemented yet");
}

extern "C" void
_mlir_ciface_transpose_3x5x4_to_5x3x4(StridedMemRefType<float, 3> *S,
                                      StridedMemRefType<float, 3> *D, int *perm,
                                      int s) {
  transposeBlas(S, D, perm, s);
}

extern "C" void
_mlir_ciface_reshape_2x3x4_to_2x12(StridedMemRefType<float, 3> *S,
                                   StridedMemRefType<float, 2> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * S->sizes[2] * sizeof(float));
}

extern "C" void
_mlir_ciface_reshape_5x3x4_to_5x12(StridedMemRefType<float, 3> *S,
                                   StridedMemRefType<float, 2> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * S->sizes[2] * sizeof(float));
}

extern "C" void
_mlir_ciface_reshape_2x12_to_2x3x4(StridedMemRefType<float, 2> *S,
                                   StridedMemRefType<float, 3> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * sizeof(float));
}

extern "C" void
_mlir_ciface_linalg_fill_view2x3xf32_f32(StridedMemRefType<float, 2> *X,
                                         float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view2x4x5xf32_f32(StridedMemRefType<float, 3> *X,
                                           float f) {
  _mlir_ciface_linalg_fill_viewsxsxsxf32_f32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view5x3x4xf32_f32(StridedMemRefType<float, 3> *X,
                                           float f) {
  _mlir_ciface_linalg_fill_viewsxsxsxf32_f32_f32(X, f);
}

extern "C" void
_mlir_ciface_transpose_5x3x4_to_4x5x3(StridedMemRefType<float, 3> *S,
                                      StridedMemRefType<float, 3> *D, int *perm,
                                      int s) {
  transposeBlas(S, D, perm, s);
}

extern "C" void
_mlir_ciface_reshape_2x4x5_to_2x20(StridedMemRefType<float, 3> *S,
                                   StridedMemRefType<float, 2> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * S->sizes[2] * sizeof(float));
}

extern "C" void
_mlir_ciface_reshape_4x5x3_to_20x3(StridedMemRefType<float, 3> *S,
                                   StridedMemRefType<float, 2> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * S->sizes[2] * sizeof(float));
}

extern "C" void
_mlir_ciface_linalg_fill_view1024x1024xf32_f32(StridedMemRefType<float, 2> *X,
                                               float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_transpose_32x1024x32_to_32x32x1024(StridedMemRefType<float, 3> *S,
                                                StridedMemRefType<float, 3> *D,
                                                int *perm, int s) {
  transposeBlas(S, D, perm, s);
}

extern "C" void
_mlir_ciface_linalg_fill_view1024x32x32xf32_f32(StridedMemRefType<float, 3> *X,
                                                float f) {
  _mlir_ciface_linalg_fill_viewsxsxsxf32_f32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view32x1024x32xf32_f32(StridedMemRefType<float, 3> *X,
                                                float f) {
  _mlir_ciface_linalg_fill_viewsxsxsxf32_f32_f32(X, f);
}

extern "C" void
_mlir_ciface_reshape_32x32x1024_to_1024x1024(StridedMemRefType<float, 3> *S,
                                             StridedMemRefType<float, 2> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * S->sizes[2] * sizeof(float));
}

extern "C" void
_mlir_ciface_reshape_1024x32x32_to_1024x1024(StridedMemRefType<float, 3> *S,
                                             StridedMemRefType<float, 2> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * S->sizes[2] * sizeof(float));
}

extern "C" void
_mlir_ciface_reshape_1024x1024_to_32x1024x32(StridedMemRefType<float, 2> *S,
                                             StridedMemRefType<float, 3> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * sizeof(float));
}

extern "C" void
_mlir_ciface_linalg_fill_view32x32x1024xf32_f32(StridedMemRefType<float, 3> *X,
                                                float f) {
  _mlir_ciface_linalg_fill_viewsxsxsxf32_f32_f32(X, f);
}

extern "C" void
_mlir_ciface_transpose_32x1024x32_to_1024x32x32(StridedMemRefType<float, 3> *S,
                                                StridedMemRefType<float, 3> *D,
                                                int *perm, int s) {
  transposeBlas(S, D, perm, s);
}

extern "C" void
_mlir_ciface_reshape_1024x1024_to_1024x32x32(StridedMemRefType<float, 2> *S,
                                             StridedMemRefType<float, 3> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * sizeof(float));
}

extern "C" void
_mlir_ciface_transpose_32x32x1024_to_32x32x1024(StridedMemRefType<float, 3> *S,
                                                StridedMemRefType<float, 3> *D,
                                                int *perm, int s) {
  transposeBlas(S, D, perm, s);
}

extern "C" void
_mlir_ciface_reshape_1024x1024_to_32x32x1024(StridedMemRefType<float, 2> *S,
                                             StridedMemRefType<float, 3> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * sizeof(float));
}

extern "C" void
_mlir_ciface_transpose_32x32x1024_to_32x1024x32(StridedMemRefType<float, 3> *S,
                                                StridedMemRefType<float, 3> *D,
                                                int *perm, int s) {
  transposeBlas(S, D, perm, s);
}

extern "C" void _mlir_ciface_transpose_32x32x32x32_to_32x32x32x32(
    StridedMemRefType<float, 4> *S, StridedMemRefType<float, 4> *D, int *perm,
    int s) {
  transposeBlas(S, D, perm, s);
}

extern "C" void
_mlir_ciface_linalg_fill_view32x32x32x32xf32_f32(StridedMemRefType<float, 4> *X,
                                                 float f) {
  _mlir_ciface_linalg_fill_viewsxsxsxsxf32_f32_f32_f32(X, f);
}

extern "C" void
_mlir_ciface_reshape_32x32x32x32_to_1024x32x32(StridedMemRefType<float, 4> *S,
                                               StridedMemRefType<float, 3> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * S->sizes[2] * S->sizes[3] * sizeof(float));
}

extern "C" void
_mlir_ciface_reshape_1024x1024_to_32x32x32x32(StridedMemRefType<float, 2> *S,
                                              StridedMemRefType<float, 4> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * sizeof(float));
}

extern "C" void
_mlir_ciface_linalg_fill_view32x32xf32_f32(StridedMemRefType<float, 2> *X,
                                           float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view32x64xf32_f32(StridedMemRefType<float, 2> *X,
                                           float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view64x32xf32_f32(StridedMemRefType<float, 2> *X,
                                           float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view900x1200xf32_f32(StridedMemRefType<float, 2> *X,
                                              float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view1100x900xf32_f32(StridedMemRefType<float, 2> *X,
                                              float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view800x1100xf32_f32(StridedMemRefType<float, 2> *X,
                                              float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view800x1200xf32_f32(StridedMemRefType<float, 2> *X,
                                              float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view800x900xf32_f32(StridedMemRefType<float, 2> *X,
                                             float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view900x1100xf32_f32(StridedMemRefType<float, 2> *X,
                                              float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view1000x900xf32_f32(StridedMemRefType<float, 2> *X,
                                              float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view1200x1100xf32_f32(StridedMemRefType<float, 2> *X,
                                               float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view800x1000xf32_f32(StridedMemRefType<float, 2> *X,
                                              float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view2000x2000xf32_f32(StridedMemRefType<float, 2> *X,
                                               float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view2000xf32_f32(StridedMemRefType<float, 1> *X,
                                          float f) {
  _mlir_ciface_linalg_fill_viewsxf32_f32(X, f);
}

// GPU - Support
#ifdef HAS_GPU_SUPPORT
extern "C" int matmulcuBlas(float *C, float *A, float *B, int M, int N, int K) {
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  float *devPtrC;
  float *devPtrA;
  float *devPtrB;

  // allocate CUDA memory
  cudaStat = cudaMalloc((void **)&devPtrC, M * N * sizeof(float));
  if (cudaStat != cudaSuccess) {
    printf("device memory allocation failed");
    return EXIT_FAILURE;
  }

  cudaStat = cudaMalloc((void **)&devPtrA, M * K * sizeof(float));
  if (cudaStat != cudaSuccess) {
    printf("device memory allocation failed");
    cudaFree(devPtrC);
    return EXIT_FAILURE;
  }
  cudaStat = cudaMalloc((void **)&devPtrB, K * N * sizeof(float));
  if (cudaStat != cudaSuccess) {
    cudaFree(devPtrA);
    cudaFree(devPtrC);
    printf("device memory allocation failed");
    return EXIT_FAILURE;
  }
  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    printf("CUBLAS initialization failed\n");
    return EXIT_FAILURE;
  }
  // set CUDA memory
  
      stat = cublasSetMatrix (N, M, sizeof(float), C, N, devPtrC, N);
      if (stat != CUBLAS_STATUS_SUCCESS) {
          printf ("data download failed");
          cudaFree (devPtrC);
          cublasDestroy(handle);
          return EXIT_FAILURE;
      }
  
  stat = cublasSetMatrix(K, M, sizeof(float), A, K, devPtrA, K);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data download failed");
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }

  stat = cublasSetMatrix(N, K, sizeof(float), B, N, devPtrB, N);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data download failed");
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }

  float alpha = 1.0f;
  // modify (handle, cublasOperation_t::CUBLAS_OP_T,
  // cublasOperation_t::CUBLAS_OP_T, devPtrA, M, N, K, 1.0f, devPtrA, K,
  // devPtrB, N, 1.0f,devPtrC, M);
  // M?
  stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, devPtrB,
                     N, devPtrA, K, &alpha, devPtrC, N);
  // Check for any errors launching the kernel
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("gemm failed");
    printf("\n %d \n", stat);
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  stat = cublasGetMatrix(N, M, sizeof(float), devPtrC, N, C, N);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data upload failed");
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  cudaFree(devPtrC);
  cudaFree(devPtrA);
  cudaFree(devPtrB);
  cublasDestroy(handle);
  return EXIT_SUCCESS;
}

int matveccuBlas( float *A, float *B, float *C, int M, int N, int K) {
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  float *devPtrC;
  float *devPtrA;
  float *devPtrB;

  // allocate CUDA memory
  cudaStat = cudaMalloc((void **)&devPtrC, M * sizeof(float));
  if (cudaStat != cudaSuccess) {
    printf("device memory allocation failed");
    return EXIT_FAILURE;
  }

  cudaStat = cudaMalloc((void **)&devPtrA, M * N * sizeof(float));
  if (cudaStat != cudaSuccess) {
    printf("device memory allocation failed");
    cudaFree(devPtrC);
    return EXIT_FAILURE;
  }
  cudaStat = cudaMalloc((void **)&devPtrB, N * sizeof(float));
  if (cudaStat != cudaSuccess) {
    cudaFree(devPtrA);
    cudaFree(devPtrC);
    printf("device memory allocation failed");
    return EXIT_FAILURE;
  }
  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    printf("CUBLAS initialization failed\n");
    return EXIT_FAILURE;
  }
  // set CUDA memory
  stat = cublasSetMatrix(N, M, sizeof(float), A, N, devPtrA, N);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data download failed");
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }

  stat = cublasSetMatrix(1, N, sizeof(float), B, 1, devPtrB, 1);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data download failed");
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  stat = cublasSetMatrix(1, M, sizeof(float), C, 1, devPtrC, 1);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data download failed");
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  float alpha = 1.0f;
  // modify (handle, cublasOperation_t::CUBLAS_OP_T,
  // cublasOperation_t::CUBLAS_OP_T, devPtrA, M, N, K, 1.0f, devPtrA, K,
  // devPtrB, N, 1.0f,devPtrC, M);
  // M?
  stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, M, N, &alpha, devPtrB,
                     1, devPtrA, N, &alpha, devPtrC, 1);
  // Check for any errors launching the kernel
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("gemm failed");
    printf("\n %d \n", stat);
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  stat = cublasGetMatrix(1, M, sizeof(float), devPtrC, 1, C, 1);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data upload failed");
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  cudaFree(devPtrC);
  cudaFree(devPtrA);
  cudaFree(devPtrB);
  cublasDestroy(handle);
  return EXIT_SUCCESS;
}




int vecmatcuBlas( float *A, float *B, float *C, int M, int N, int K) {
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  float *devPtrC;
  float *devPtrA;
  float *devPtrB;

  // allocate CUDA memory
  cudaStat = cudaMalloc((void **)&devPtrC, N * sizeof(float));
  if (cudaStat != cudaSuccess) {
    printf("device memory allocation failed");
    return EXIT_FAILURE;
  }

  cudaStat = cudaMalloc((void **)&devPtrA, M * N * sizeof(float));
  if (cudaStat != cudaSuccess) {
    printf("device memory allocation failed");
    cudaFree(devPtrC);
    return EXIT_FAILURE;
  }
  cudaStat = cudaMalloc((void **)&devPtrB, M * sizeof(float));
  if (cudaStat != cudaSuccess) {
    cudaFree(devPtrA);
    cudaFree(devPtrC);
    printf("device memory allocation failed");
    return EXIT_FAILURE;
  }
  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    printf("CUBLAS initialization failed\n");
    return EXIT_FAILURE;
  }
  // set CUDA memory
  stat = cublasSetMatrix(N, M, sizeof(float), A, N, devPtrA, N);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data download failed");
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }

  stat = cublasSetMatrix(M, K, sizeof(float), B, M, devPtrB, M);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data download failed");
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  stat = cublasSetMatrix(N, K, sizeof(float), C, N, devPtrC, N);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data download failed");
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  float alpha = 1.0f;
  // modify (handle, cublasOperation_t::CUBLAS_OP_T,
  // cublasOperation_t::CUBLAS_OP_T, devPtrA, M, N, K, 1.0f, devPtrA, K,
  // devPtrB, N, 1.0f,devPtrC, M);
  // M?
  stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, K, M, &alpha, devPtrA,
                     N, devPtrB, M, &alpha, devPtrC, N);
  // Check for any errors launching the kernel
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("gemm failed");
    printf("\n %d \n", stat);
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  stat = cublasGetMatrix(N, K, sizeof(float), devPtrC, N, C, N);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data upload failed");
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  cudaFree(devPtrC);
  cudaFree(devPtrA);
  cudaFree(devPtrB);
  cublasDestroy(handle);
  return EXIT_SUCCESS;
}
#endif
