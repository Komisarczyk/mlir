#define M 1000
#define N 1200
float A[M][M];
float B[M][N];
int main(){
float alpha, beta;
#pragma scop
  for (int i = 0; i < M; i++)
     for (int j = 0; j < N; j++) {
        for (int k = i + 1; k < M; k++)
           B[i][j] += A[k][i] * B[k][j];
        B[i][j] = alpha * B[i][j];
     }
#pragma endscop
return 0;
}
