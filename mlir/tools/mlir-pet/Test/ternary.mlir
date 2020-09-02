// RUN: mlir-pet %S/Inputs/ternary.c | FileCheck %s


CHECK:  func @scop_entry(%arg{{.*}}: memref<1xf32>, %arg{{.*}}: memref<1xf32>) {
CHECK:    %{{.*}} = alloc() : memref<1xf32>
CHECK:    %{{.*}} = constant 1.000000e+00 : f32
CHECK:    %{{.*}} = constant 0 : index
CHECK:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1xf32>
CHECK:    %{{.*}} = alloc() : memref<1xf32>
CHECK:    %{{.*}} = constant 0 : index
CHECK:    %{{.*}} = affine.load %{{.*}}[%{{.*}}] : memref<1xf32>
CHECK:    %{{.*}} = constant 0.000000e+00 : f32
CHECK:    %{{.*}} = cmpf "ogt", %{{.*}}, %{{.*}} : f32
CHECK:    %{{.*}} = constant 0 : index
CHECK:    %{{.*}} = affine.load %arg{{.*}}[%{{.*}}] : memref<1xf32>
CHECK:    %{{.*}} = constant 0 : index
CHECK:    %{{.*}} = affine.load %arg{{.*}}[%{{.*}}] : memref<1xf32>
CHECK:    %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : f32
CHECK:    %{{.*}} = constant 0 : index
CHECK:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1xf32>


