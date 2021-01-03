float A[150][150];
float x[150];
float y[150];
float tmp[150];

int main(){



#pragma scop
  for (int i = 0; i < 150; i++)
    y[i] = 0.0;
  for (int i = 0; i < 150; i++)
    {
      tmp[i] = 0.0f;
      for (int j = 0; j < 150; j++)
	tmp[i] = tmp[i] + A[i][j] * x[j];
      for (int j = 0; j < 150; j++)
	y[j] = y[j] + A[i][j] * tmp[i];
    }
#pragma endscop
return 0;
}
