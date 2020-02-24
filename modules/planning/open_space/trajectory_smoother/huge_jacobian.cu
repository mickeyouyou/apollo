#include <adolc/adouble.h>
#include <adolc/adoublecuda.h>
#include <chrono>
#include <iostream>

__global__ void jacobi(float *x, float *y, float *deriv, long n) {
  long id = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = threadIdx.y;
  if (id < n && idy < 128) {
    adtlc::adouble ax[128], ay[5];
    for (int i = 0; i < 128; i++) {
      ax[i] = x[128 * id + i];
      ax[idy].setADValue(1);
      ay[i] = 0;
    }

    for (int i = 0; i < 128; i++) {
      ay[0] += ax[i];
      ay[1] += pow(ax[i], 2.0);
      ay[2] += pow(ax[i], 3.0);
      ay[3] += pow(ax[i], 4.0);
      ay[4] += pow(ax[i], 5.0);
    }
    for (int i = 0; i < 5; i++) {
      deriv[5 * 128 * id + 5 * idy + i] = (float)(ay[i].getADValue());
    }
    if (idy == 0)
      for (int i = 0; i < 5; i++) y[5 * id + i] = (float)(ay[i].getValue());
  }
}

int main() {
  auto t1 = std::chrono::system_clock::now();
  long N = 1 << 5;
  cout << "Number of matrix: " << N << std::endl;
  float x[128 * N], y[5 * N], deriv[5 * 128 * N];
  float *d_x, *d_y, *d_deriv;
  int size = sizeof(float);
  for (long i = 0; i < 128 * N; i++) {
    x[i] = i;
  }
  cudaMalloc((void **)&d_x, 128 * N * size);
  cudaMalloc((void **)&d_y, 5 * N * size);
  cudaMalloc((void **)&d_deriv, 128 * 5 * N * size);
  cudaMemcpy(d_x, &x, 128 * N * size, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(4, 128);
  jacobi<<<N / 4, threadsPerBlock>>>(d_x, d_y, d_deriv, N);
  cudaError_t error;
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("Code: %d, Message: %s\n", error, cudaGetErrorString(error));
  }

  cudaMemcpy(&y, d_y, 5 * N * size, cudaMemcpyDeviceToHost);
  cudaMemcpy(&deriv, d_deriv, 128 * 5 * N * size, cudaMemcpyDeviceToHost);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_deriv);

  auto t2 = std::chrono::system_clock::now();
  auto duration = std::chrono::duration<double>(t2 - t1);

  std::cout << "Total time on CUDA(milli secs):" << duration.count() * 1000
            << std::endl;

  //   for (long i = 0; i < 5; i++) {
  //     std::cout << "Result y0 - y4" << std::endl;
  //     for (int h = 0; h < 5; h++) std::cout << y[5 * i + h] << " ";
  //     std::cout << std::endl << "Jacobi matrix:" << std::endl;
  //     for (int j = 0; j < 128; j++) {
  //       for (int k = 0; k < 5; k++)
  //         std::cout << deriv[5 * 128 * i + 5 * j + k] << " ";
  //       std::cout << std::endl;
  //     }
  //     std::cout << std::endl;
  //   }

  return 0;
}