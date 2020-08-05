float A[10][30];
float B[30][5];
float C[30][60];
float D[5][60];
float E[10][60];
void print_memref_f32(float a);
// float beta = 1.0;
float alpha = 1.0;
/*
int main(void) {

#pragma scop
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 5; j++){
      for (int k = 0; k < 30; k++) {
          C[i][j] += alpha * A[i][k] * B[k][j];
      }
    }
  }
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 60; j++){
        for (int k = 0; k < 5; k++) {
          E[i][j] += alpha * C[i][k] * D[k][j];
      }
    }
  }
#pragma endscop

  return 0;
}
*/
int main(void) {

#pragma scop

  for (int i = 0; i < 30; i++) {
    for (int j = 0; j < 60; j++) {
      for (int k = 0; k < 5; k++) {
        C[i][j] += alpha * B[i][k] * D[k][j];
      }
    }
  }
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 60; j++) {
      for (int k = 0; k < 30; k++) {
        E[i][j] += alpha * A[i][k] * C[k][j];
      }
    }
  }

#pragma endscop

  return 0;
}
