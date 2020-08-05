float A[100][100];
float B[100][100];
float C[100][100];
int main() {
#pragma scop

  for (int i = 0; i < 100; i++)
    for (int j = 0; j < 90; j++) {
      B[j][i] = A[i][j];
    }
  for (int i = 0; i < 90; i++)
    for (int j = 0; j < 100; j++) {
      A[j][i] = B[i][j];
    }
  for (int i = 0; i < 100; i++)
    for (int j = 0; j < 90; j++) {
      C[j][i] = A[i][j];
    }


#pragma endscop
  return 0;
}
