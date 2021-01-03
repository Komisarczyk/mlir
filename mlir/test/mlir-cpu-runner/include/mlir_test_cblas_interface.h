//===- mlir_test_cblas_interface.h - Simple Blas subset interface ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CPU_RUNNER_MLIR_TEST_CBLAS_INTERFACE_H_
#define MLIR_CPU_RUNNER_MLIR_TEST_CBLAS_INTERFACE_H_
#include <mkl.h>
#include "mlir/ExecutionEngine/RunnerUtils.h"
#include <chrono>
#ifdef _WIN32
#ifndef MLIR_TEST_CBLAS_INTERFACE_EXPORT
#ifdef mlir_test_cblas_interface_EXPORTS
/* We are building this library */
#define MLIR_TEST_CBLAS_INTERFACE_EXPORT __declspec(dllexport)
#else
/* We are using this library */
#define MLIR_TEST_CBLAS_INTERFACE_EXPORT __declspec(dllimport)
#endif // mlir_test_cblas_interface_EXPORTS
#endif // MLIR_TEST_CBLAS_INTERFACE_EXPORT
#else
#define MLIR_TEST_CBLAS_INTERFACE_EXPORT
#endif // _WIN32

class Timer {
private:
  static Timer *timer_instance;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  float s_initial;

public:
  void start_timer() { start = std::chrono::system_clock::now(); 
   s_initial = dsecnd();
  
  }

  void stop_timer() {
    end = std::chrono::system_clock::now();
    std::cout << (dsecnd() - s_initial);
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << std::endl;
  }

  static Timer *get_instance() {
    if (!timer_instance)
      timer_instance = new Timer();
    return timer_instance;
  }
};
int matveccuBlas(float *C, float *A, float *B, int M, int N, int K);
int vecmatcuBlas( float *A, float *B, float *C, int M, int N, int K);
extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void start_timer();

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void stop_timer();

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_viewf32_f32(StridedMemRefType<float, 0> *X, float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_viewsxf32_f32(StridedMemRefType<float, 1> *X, float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_viewsxsxf32_f32(StridedMemRefType<float, 2> *X,
                                         float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_viewsxsxsxf32_f32_f32(StridedMemRefType<float, 3> *X,
                                               float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_copy_viewf32_viewf32(StridedMemRefType<float, 0> *I,
                                         StridedMemRefType<float, 0> *O);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_copy_viewsxf32_viewsxf32(StridedMemRefType<float, 1> *I,
                                             StridedMemRefType<float, 1> *O);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_copy_viewsxsxf32_viewsxsxf32(
    StridedMemRefType<float, 2> *I, StridedMemRefType<float, 2> *O);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_dot_viewsxf32_viewsxf32_viewf32(
    StridedMemRefType<float, 1> *X, StridedMemRefType<float, 1> *Y,
    StridedMemRefType<float, 0> *Z);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
cublasMatmul(
    float *C_allocatedptr, float *C_alignedptr,
    int64_t C_offset, int64_t C_sizes0, int64_t C_sizes1, int64_t C_strides0,
    int64_t C_strides1, float *A_allocatedptr, float *A_alignedptr,
    int64_t A_offset, int64_t A_sizes0, int64_t A_sizes1, int64_t A_strides0,
    int64_t A_strides1, float *B_allocatedptr, float *B_alignedptr,
    int64_t B_offset, int64_t B_sizes0, int64_t B_sizes1, int64_t B_strides0,
    int64_t B_strides1);


extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
cublasMatvec(float *C_allocatedptr, float *C_alignedptr, int64_t C_offset,
             int64_t C_sizes0, int64_t C_sizes1, int64_t C_strides0,
             int64_t C_strides1, float *A_allocatedptr, float *A_alignedptr,
             int64_t A_offset, int64_t A_sizes0, int64_t A_strides0,
             float *B_allocatedptr, float *B_alignedptr, int64_t B_offset,
             int64_t B_sizes0, int64_t B_strides0);



extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_matvec_2000x2000x2000(StridedMemRefType<float, 1> *x,
                                   StridedMemRefType<float, 2> *A,
                                   StridedMemRefType<float, 1> *y, float alpha,
                                   float beta, int transA);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view42x42xf32_f32(StridedMemRefType<float, 2> *X,
                                           float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view3x5x4xf32_f32(StridedMemRefType<float, 3> *X,
                                           float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view2x3x4xf32_f32(StridedMemRefType<float, 3> *X,
                                           float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view2x5xf32_f32(StridedMemRefType<float, 2> *X,
                                         float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_transpose_3x5x4_to_5x3x4(StridedMemRefType<float, 3> *S,
                                      StridedMemRefType<float, 3> *D, int *perm,
                                      int s);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_reshape_2x12_to_2x3x4(StridedMemRefType<float, 2> *S,
                                   StridedMemRefType<float, 3> *D);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_reshape_5x3x4_to_5x12(StridedMemRefType<float, 3> *S,
                                   StridedMemRefType<float, 2> *D);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_reshape_2x3x4_to_2x12(StridedMemRefType<float, 3> *S,
                                   StridedMemRefType<float, 2> *D);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view2x3xf32_f32(StridedMemRefType<float, 2> *X,
                                         float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view2x4x5xf32_f32(StridedMemRefType<float, 3> *X,
                                           float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view5x3x4xf32_f32(StridedMemRefType<float, 3> *X,
                                           float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_transpose_5x3x4_to_4x5x3(StridedMemRefType<float, 3> *S,
                                      StridedMemRefType<float, 3> *D, int *perm,
                                      int s);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_reshape_2x4x5_to_2x20(StridedMemRefType<float, 3> *S,
                                   StridedMemRefType<float, 2> *D);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_reshape_4x5x3_to_20x3(StridedMemRefType<float, 3> *S,
                                   StridedMemRefType<float, 2> *D);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view1024x1024xf32_f32(StridedMemRefType<float, 2> *X,
                                               float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_transpose_32x1024x32_to_32x32x1024(StridedMemRefType<float, 3> *S,
                                                StridedMemRefType<float, 3> *D,
                                                int *perm, int s);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view1024x32x32xf32_f32(StridedMemRefType<float, 3> *X,
                                                float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view32x1024x32xf32_f32(StridedMemRefType<float, 3> *X,
                                                float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_reshape_32x32x1024_to_1024x1024(StridedMemRefType<float, 3> *S,
                                             StridedMemRefType<float, 2> *D);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_reshape_1024x32x32_to_1024x1024(StridedMemRefType<float, 3> *S,
                                             StridedMemRefType<float, 2> *D);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_reshape_1024x1024_to_32x1024x32(StridedMemRefType<float, 2> *S,
                                             StridedMemRefType<float, 3> *D);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view32x32x1024xf32_f32(StridedMemRefType<float, 3> *X,
                                                float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_transpose_32x1024x32_to_1024x32x32(StridedMemRefType<float, 3> *S,
                                                StridedMemRefType<float, 3> *D,
                                                int *perm, int s);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_reshape_1024x1024_to_1024x32x32(StridedMemRefType<float, 2> *S,
                                             StridedMemRefType<float, 3> *D);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_transpose_32x32x1024_to_32x32x1024(StridedMemRefType<float, 3> *S,
                                                StridedMemRefType<float, 3> *D,
                                                int *perm, int s);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_reshape_1024x1024_to_32x32x1024(StridedMemRefType<float, 2> *S,
                                             StridedMemRefType<float, 3> *D);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_transpose_32x32x1024_to_32x1024x32(StridedMemRefType<float, 3> *S,
                                                StridedMemRefType<float, 3> *D,
                                                int *perm, int s);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_transpose_32x32x32x32_to_32x32x32x32(
    StridedMemRefType<float, 4> *S, StridedMemRefType<float, 4> *D, int *perm,
    int s);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view32x32x32x32xf32_f32(StridedMemRefType<float, 4> *X,
                                                 float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_reshape_32x32x32x32_to_1024x32x32(StridedMemRefType<float, 4> *S,
                                               StridedMemRefType<float, 3> *D);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_reshape_1024x1024_to_32x32x32x32(StridedMemRefType<float, 2> *S,
                                              StridedMemRefType<float, 4> *D);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view32x32xf32_f32(StridedMemRefType<float, 2> *X,
                                           float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view32x64xf32_f32(StridedMemRefType<float, 2> *X,
                                           float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view64x32xf32_f32(StridedMemRefType<float, 2> *X,
                                           float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view900x1200xf32_f32(StridedMemRefType<float, 2> *X,
                                              float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view1100x900xf32_f32(StridedMemRefType<float, 2> *X,
                                              float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view800x1100xf32_f32(StridedMemRefType<float, 2> *X,
                                              float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view800x1200xf32_f32(StridedMemRefType<float, 2> *X,
                                              float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view800x900xf32_f32(StridedMemRefType<float, 2> *X,
                                             float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view900x1100xf32_f32(StridedMemRefType<float, 2> *X,
                                              float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view1000x900xf32_f32(StridedMemRefType<float, 2> *X,
                                              float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view1200x1100xf32_f32(StridedMemRefType<float, 2> *X,
                                               float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view800x1000xf32_f32(StridedMemRefType<float, 2> *X,
                                              float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view1200x1200xf32_f32(StridedMemRefType<float, 2> *X,
                                               float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view1200x1000xf32_f32(StridedMemRefType<float, 2> *X,
                                               float f);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_linalg_fill_view2000xf32_f32(StridedMemRefType<float, 1> *X,
                                          float f);
extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT int matmulcuBlas(float* C,
                float* A, float* B,int M, int N, int K);
#ifdef HAS_GPU_SUPPORT
extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void *
_mlir_ciface_allocateMemoryForDevice(int64_t size);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_createCallCopyFromHostToDevice(StridedMemRefType<float, 2> *S,
                                            void *D, int64_t size);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_createCallToCublasSgemm(void *C, void *A, void *B,
                                     StridedMemRefType<float, 2> *CMemref,
                                     StridedMemRefType<float, 2> *AMemref,
                                     StridedMemRefType<float, 2> *BMemref);

extern "C" MLIR_TEST_CBLAS_INTERFACE_EXPORT void
_mlir_ciface_createCallCopyFromDeviceToHost(void *S,
                                            StridedMemRefType<float, 2> *D,
                                            int64_t size);
#endif // HAS_GPU_SUPPORT

#endif // MLIR_CPU_RUNNER_CBLAS_INTERFACE_H_
