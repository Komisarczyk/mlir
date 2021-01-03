float A[150][150];
float x1[150];
float y1[150];
float x2[150];
float y2[150];


int main(void) {

#pragma scop
  for (int i = 0; i < 150; i++)
    for (int j = 0; j < 150; j++)
      x1[i] = x1[i] + A[i][j] * y1[j];
  for (int i = 0; i < 150; i++)
    for (int j = 0; j < 150; j++)
      x2[i] = x2[i] + A[j][i] * y2[j];
#pragma endscop

  return 0;
}