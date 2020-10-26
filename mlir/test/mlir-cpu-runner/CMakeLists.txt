set(LLVM_OPTIONAL_SOURCES
  mlir_test_cblas.cpp
  mlir_test_cblas_interface.cpp
  )

add_llvm_library(mlir_test_cblas SHARED mlir_test_cblas.cpp)
target_compile_definitions(mlir_test_cblas PRIVATE mlir_test_cblas_EXPORTS)

find_library(dnnlPATH libdnnl.so.1)
find_library(cudartPATH libcudart.so)
find_library(cublasPATH libcublas.so)
find_path(cudaInclude cublas_v2.h)

#add_llvm_library(mlir_test_cblas_interface SHARED mlir_test_cblas_interface.cpp "/usr/local/cuda-10.2/targets/x86_64-linux/include/cublas_v2.h" "/usr/local/cuda-10.2/targets/x86_64-linux/include/cuda_runtime.h")
add_llvm_library(mlir_test_cblas_interface SHARED mlir_test_cblas_interface.cpp)
target_include_directories(mlir_test_cblas_interface PRIVATE "${cudaInclude}")
#target_link_libraries(mlir_test_cblas_interface PRIVATE mlir_test_cblas libdnnl.so cublas.so cudart.so PUBLIC "${dnnlPATH}" "/usr/local/cuda-10.2/targets/x86_64-linux/lib/libcublas.so.10" "/usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudart.so.10.2")
target_link_libraries(mlir_test_cblas_interface PRIVATE mlir_test_cblas libdnnl.so cublas.so cudart.so PUBLIC "${dnnlPATH}" "${cublasPATH}" "${cudartPATH}" )
target_compile_definitions(mlir_test_cblas_interface PRIVATE mlir_test_cblas_interface_EXPORTS)
