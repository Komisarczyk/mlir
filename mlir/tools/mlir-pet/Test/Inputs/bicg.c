float A[150][100];
float s[100];
float q[150];
float p[100];
float r[150];

int main(){
float alpha, beta;
#pragma scop
  for (int i = 0; i < 100; i++)
    s[i] = 0.0;
  for (int i = 0; i < 150; i++)
    {
      q[i] = 0.0f;
      for (int j = 0; j < 100; j++)
	{
	  s[j] = s[j] + r[i] * A[i][j];
	  q[i] = q[i] + A[i][j] * p[j];
	}
    }
#pragma endscop
return 0;
}
