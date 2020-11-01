set(LLVM_OPTIONAL_SOURCES
  mlir_test_cblas.cpp
  mlir_test_cblas_interface.cpp
  )

add_llvm_library(mlir_test_cblas SHARED mlir_test_cblas.cpp)
target_compile_definitions(mlir_test_cblas PRIVATE mlir_test_cblas_EXPORTS)

#find_library(dnnlPATH libdnnl.so.1)
find_library(mklPATH libmkl_rt.so)
find_library(cudartPATH libcudart.so)
find_library(cublasPATH libcublas.so)
find_path(cudaInclude cublas_v2.h)
find_path(mklInclude mkl.h)

add_llvm_library(mlir_test_cblas_interface SHARED mlir_test_cblas_interface.cpp)
target_include_directories(mlir_test_cblas_interface PRIVATE "${cudaInclude}" "${mklInclude}")
target_link_libraries(mlir_test_cblas_interface PRIVATE mlir_test_cblas libdnnl.so cublas.so cudart.so PUBLIC "${mklPATH}" "${cublasPATH}" "${cudartPATH}" )
target_compile_definitions(mlir_test_cblas_interface PRIVATE mlir_test_cblas_interface_EXPORTS)