float A[800][1100];
float B[1100][900];
float C[800][900];
float D[900][1200];
float E[800][1200];
float F[1200][100];
float G[800][100];
float beta = 1.0f;
float alpha = 1.0f;

int main(void) {

#pragma scop
  for (int i = 0; i < 800; i++) {
    for (int j = 0; j < 900; j++)
      C[i][j] *= beta;
    for (int k = 0; k < 1100; k++) {
      for (int j = 0; j < 900; j++)
        C[i][j] += alpha* A[i][k] *  B[k][j];
    }
  }
    for (int i = 0; i < 800; i++) {
    for (int j = 0; j < 1200; j++)
      E[i][j] *= beta;
    for (int k = 0; k < 900; k++) {
      for (int j = 0; j < 1200; j++)
        E[i][j] += alpha* C[i][k] *   D[k][j];
    }
  }
      for (int i = 0; i < 800; i++) {
    for (int j = 0; j < 100; j++)
      G[i][j] *= beta;
    for (int k = 0; k < 1200; k++) {
      for (int j = 0; j < 100; j++)
      G[i][j] += alpha* E[i][k] * F[k][j] ;
    }
  }
#pragma endscop

  return 0;
}