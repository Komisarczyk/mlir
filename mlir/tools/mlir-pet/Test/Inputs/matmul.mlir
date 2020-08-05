#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<() -> (100)>
#map4 = affine_map<() -> (800)>
#map5 = affine_map<() -> (1200)>
#map6 = affine_map<() -> (900)>


module {
  func @scop_entry(%arg0: memref<800x1100xf32>, %arg1: memref<1100x900xf32>, %arg2: memref<800x900xf32>, %arg3: memref<900x1200xf32>, %arg4: memref<800x1200xf32>, %arg5: memref<1200x100xf32>, %arg6: memref<800x100xf32>, %arg7: memref<1xf32>, %arg8: memref<1xf32>) {
    affine.for %arg9 = 0 to 800 {
      affine.for %arg10 = 0 to 100 {
        %c0 = constant 0 : index
        %0 = affine.load %arg8[%c0] : memref<1xf32>
        %1 = affine.load %arg6[%arg9, %arg10] : memref<800x100xf32>
        %2 = mulf %0, %1 : f32
        affine.store %2, %arg6[%arg9, %arg10] : memref<800x100xf32>
      }
    }
    affine.for %arg9 = 0 to 800 {
      affine.for %arg10 = 0 to 1200 {
        %c0 = constant 0 : index
        %0 = affine.load %arg8[%c0] : memref<1xf32>
        %1 = affine.load %arg4[%arg9, %arg10] : memref<800x1200xf32>
        %2 = mulf %0, %1 : f32
        affine.store %2, %arg4[%arg9, %arg10] : memref<800x1200xf32>
      }
    }
    affine.for %arg9 = 0 to 800 {
      affine.for %arg10 = 0 to 900 {
        %c0 = constant 0 : index
        %0 = affine.load %arg8[%c0] : memref<1xf32>
        %1 = affine.load %arg2[%arg9, %arg10] : memref<800x900xf32>
        %2 = mulf %0, %1 : f32
        affine.store %2, %arg2[%arg9, %arg10] : memref<800x900xf32>
      }
    }
    linalg.matmul(%arg0, %arg1, %arg2) : memref<800x1100xf32>, memref<1100x900xf32>, memref<800x900xf32>
    linalg.matmul(%arg2, %arg3, %arg4) : memref<800x900xf32>, memref<900x1200xf32>, memref<800x1200xf32>
    linalg.matmul(%arg4, %arg5, %arg6) : memref<800x1200xf32>, memref<1200x100xf32>, memref<800x100xf32>
    return
  }
}