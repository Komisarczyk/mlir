func @chainMatmul() {
    %arg0 = alloc() : memref<800x1100xf32>
    %arg1 = alloc() : memref<1100x900xf32>
    %arg2 = alloc() : memref<900x1200xf32>
    %arg3 = alloc() : memref<1200x100xf32>
    %arg4 = alloc() : memref<800x900xf32>
    %arg5 = alloc() : memref<800x1200xf32>
    %arg6 = alloc() : memref<800x100xf32>

    linalg.matmul( %arg0, %arg1, %arg4) :
     memref<800x1100xf32>, memref<1100x900xf32>, memref<800x900xf32>
    linalg.matmul( %arg4, %arg2, %arg5) :
     memref<800x900xf32>, memref<900x1200xf32>, memref<800x1200xf32>
    linalg.matmul( %arg5, %arg3, %arg6) :
      memref<800x1200xf32>, memref<1200x100xf32>, memref<800x100xf32>
    return
}
