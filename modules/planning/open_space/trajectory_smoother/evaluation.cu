/******************************************************************************
 * Copyright 2019 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#include "evaluation.h"

#include <thrust/device_vector.h>

namespace apollo {
namespace planning {
bool InitialCuda() {
  int dev = 0;
  cudaDeviceProp deviceProp;
  checkCuda(cudaGetDeviceProperties(&deviceProp, dev));
  printf("Using device %d: %s\n", dev, deviceProp.name);
  checkCuda(cudaSetDevice(dev));
  return true;
}

__global__ void data_feed_kernel(int *iRow, int *jCol, unsigned int *rind_L,
                                 unsigned int *cind_L, const int nnz_L) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;

  if (index < nnz_L) {
    iRow[index] = rind_L[index];
    jCol[index] = cind_L[index];
    // printf("data_feed_kernel rind_L[%d]:%d, cind_L[%d]:%d \n", index,
    //        rind_L[index], index, cind_L[index]);
  }
}

__global__ void set_kernel(double *dst, const double *src, const int size) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;

  if (index < size) {
    dst[index] = src[index];
  }
}

void data_feed_util(int *iRow, int *jCol, unsigned int *rind_L,
                    unsigned int *cind_L, const int nnz_L) {
  // InitialCuda();
  printf("data_feed_util nnz_L: %d \n", nnz_L);
  size_t threads_per_block = 1024;
  size_t number_of_blocks = (nnz_L + threads_per_block - 1) / threads_per_block;

  int *d_iRow, *d_jCol;
  unsigned int *d_rind_L, *d_cind_L;

  unsigned int nBytes = nnz_L * sizeof(int);
  unsigned int nUBytes = nnz_L * sizeof(unsigned int);
  cudaMalloc((void **)&d_iRow, nBytes);
  cudaMalloc((void **)&d_jCol, nBytes);
  cudaMalloc((void **)&d_rind_L, nUBytes);
  cudaMalloc((void **)&d_cind_L, nUBytes);

  cudaMemcpy(d_iRow, iRow, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_jCol, jCol, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_rind_L, rind_L, nUBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_cind_L, cind_L, nUBytes, cudaMemcpyHostToDevice);

  // thrust::device_vector<int> d_iRow(iRow, iRow + nnz_L);
  // thrust::device_vector<int> d_jCol(jCol, jCol + nnz_L);

  // thrust::device_vector<unsigned int> d_rind_L(rind_L, rind_L + nnz_L);
  // thrust::device_vector<unsigned int> d_cind_L(cind_L, cind_L + nnz_L);

  cudaError_t error;
  data_feed_kernel<<<number_of_blocks, threads_per_block>>>(
      d_iRow, d_jCol, d_rind_L, d_cind_L, nnz_L);
  // thrust::host_vector<int> h_iRow(d_iRow.begin(), d_iRow.end() + nnz_L);
  // thrust::host_vector<int> h_jCol(d_jCol.begin(), d_jCol.end() + nnz_L);

  error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("Code: %d, Message: %s\n", error, cudaGetErrorString(error));
  }
  checkCuda(cudaDeviceSynchronize());
  cudaMemcpy(iRow, d_iRow, nBytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(jCol, d_jCol, nBytes, cudaMemcpyDeviceToHost);

  cudaFree(d_iRow);
  cudaFree(d_jCol);
  cudaFree(d_rind_L);
  cudaFree(d_cind_L);
  cudaDeviceReset();
}

void data_set_util(double *dst, const double *src, const int size) {
  int threads_per_block = 512;
  int blocks = (size + threads_per_block - 1) / threads_per_block;

  double *d_dst, *d_src;

  unsigned int nBytes = size * sizeof(double);
  cudaMalloc(&d_dst, nBytes);
  cudaMalloc(&d_src, nBytes);

  cudaMemcpy(d_dst, dst, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_src, src, nBytes, cudaMemcpyHostToDevice);

  set_kernel<<<blocks, threads_per_block>>>(d_dst, d_src, size);

  cudaError_t error;
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("Code: %d, Message: %s\n", error, cudaGetErrorString(error));
  }
  checkCuda(cudaDeviceSynchronize());
  // cudaMemcpy(src, d_src, nBytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(dst, d_dst, nBytes, cudaMemcpyDeviceToHost);
  cudaFree(d_dst);
  cudaFree(d_src);
}

// template <typename T>
__global__ void kernel_objective(int n, const double *x, double ts_,
                                 int horizon_, double *last_time_u_,
                                 double *xWS_, double *xf_, int obstacles_num_,
                                 int obstacles_edges_sum_, double *obj_value) {
  // penalty
  //   maybe use thrust to keep these weights
  double weight_state_x_ = 18.0;
  double weight_state_y_ = 14.0;
  double weight_state_phi_ = 10.0;
  double weight_state_v_ = 0.0;
  double weight_input_steer_ = 0.3;
  double weight_input_a_ = 1.1;
  double weight_rate_steer_ = 2.0;
  double weight_rate_a_ = 2.5;
  double weight_stitching_steer_ = 1.75;
  double weight_stitching_a_ = 3.25;
  double weight_first_order_time_ = 1.0;
  double weight_second_order_time_ = 2.0;
  double weight_end_state_ = 1.0;
  double weight_slack_ = 1.0;

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  int state_start_index_ = 0;
  int control_start_index_ = 4 * (horizon_ + 1);
  int time_start_index_ = control_start_index_ + 2 * horizon_;
  int l_start_index_ = time_start_index_ + (horizon_ + 1);
  int n_start_index_ = l_start_index_ + obstacles_edges_sum_ * (horizon_ + 1);

  int slack_horizon_ = obstacles_num_ * (horizon_ + 1);
  int slack_index_ = n_start_index_ + 4 * obstacles_num_ * (horizon_ + 1);

  // Objective is :
  // min control inputs
  // min input rate
  // min time (if the time step is not fixed)
  // regularization wrt warm start trajectory
  //   DCHECK(ts_ != 0) << "ts in distance_approach_ is 0";
  int control_index = control_start_index_;
  int time_index = time_start_index_;
  // int state_index = state_start_index_;

  *obj_value = 0.0;
  // 1. objective to minimize state diff to warm up
  //    note: CHECK_EQ(horizon_ + 1, xWS_.cols());
  //   for (int i = 0; i < horizon_ + 1; ++i) {
  double x1_diff = x[i * 4] - xWS_[0 * (horizon_ + 1) + i];
  double x2_diff = x[i * 4 + 1] - xWS_[1 * (horizon_ + 1) + i];
  double x3_diff = x[i * 4 + 2] - xWS_[2 * (horizon_ + 1) + i];
  double x4_abs = x[i * 4 + 3];
  *obj_value += weight_state_x_ * x1_diff * x1_diff +
                weight_state_y_ * x2_diff * x2_diff +
                weight_state_phi_ * x3_diff * x3_diff +
                weight_state_v_ * x4_abs * x4_abs;
  //   state_index += 4;
  //   }

  printf("obj_value after state diff: %f: \n", *obj_value);

  // 2. objective to minimize u square
  //   for (int i = 0; i < horizon_; ++i) {
  *obj_value += weight_input_steer_ * x[4 * (horizon_ + 1) + i * 2] *
                    x[4 * (horizon_ + 1) + i * 2] +
                weight_input_a_ * x[4 * (horizon_ + 1) + i * 2 + 1] *
                    x[4 * (horizon_ + 1) + i * 2 + 1];
  //   control_index += 2;
  //   }

  printf("obj_value after minimize u square: %f: \n", *obj_value);

  // 3. objective to minimize input change rate for first horizon
  control_index = control_start_index_;
  double last_time_steer_rate =
      (x[control_index] - last_time_u_[0]) / x[time_index] / ts_;
  double last_time_a_rate =
      (x[control_index + 1] - last_time_u_[1]) / x[time_index] / ts_;
  *obj_value +=
      weight_stitching_steer_ * last_time_steer_rate * last_time_steer_rate +
      weight_stitching_a_ * last_time_a_rate * last_time_a_rate;

  printf("obj_value after minimize input change rate: %f: \n", *obj_value);

  // 4. objective to minimize input change rates, [0- horizon_ -2]
  time_index++;
  //   for (int i = 0; i < horizon_ - 1; ++i) {
  double steering_rate =
      (x[control_index + 2] - x[control_index]) / x[time_index] / ts_;
  double a_rate =
      (x[control_index + 3] - x[control_index + 1]) / x[time_index] / ts_;
  *obj_value += weight_rate_steer_ * steering_rate * steering_rate +
                weight_rate_a_ * a_rate * a_rate;
  //   control_index += 2;
  //   time_index++;
  //   }

  printf("obj_value after minimize input change rates: %f: \n", *obj_value);

  // 5. objective to minimize total time [0, horizon_]
  time_index = time_start_index_;
  //   for (int i = 0; i < horizon_ + 1; ++i) {
  double first_order_penalty = weight_first_order_time_ * x[time_index];
  double second_order_penalty =
      weight_second_order_time_ * x[time_index] * x[time_index];
  *obj_value += first_order_penalty + second_order_penalty;
  // time_index++;
  //   }

  printf("obj_value after minimize total time: %f: \n", *obj_value);

  // 6. end state constraints
  for (int i = 0; i < 4; ++i) {
    *obj_value += weight_end_state_ *
                  (x[state_start_index_ + 4 * horizon_ + i] - xf_[i]) *
                  (x[state_start_index_ + 4 * horizon_ + i] - xf_[i]);
  }

  // 7. slack variables
  for (int i = 0; i < slack_horizon_; ++i) {
    *obj_value += weight_slack_ * x[slack_index_ + i];
  }

  printf("final obj_value after cuda evaluation: %f: \n", *obj_value);
}

// template <class T>
void evalue_objective(int n, const double *x, double ts_, int horizon_,
                      double *last_time_u_, double *xWS_, double *xf_,
                      int obstacles_num_, int obstacles_edges_sum_,
                      double *obj_value) {
  // TODO 1 blocks and threads
  int threads = 1024;
  int blocks = 2;
  printf("debug data in evalue_objective, n: %d: \n", n);
  // TODO 2 transform x from host to device
  //   COPY DATA FROM HOST TO DEVICE BY THRUST

  //   thrust::device_vector<T> D(4);
  //   T * obj_value = obj_value;
  kernel_objective<<<blocks, threads>>>(n, x, ts_, horizon_, last_time_u_, xWS_,
                                        xf_, obstacles_num_,
                                        obstacles_edges_sum_, obj_value);
  cudaDeviceSynchronize();
}

}  // namespace planning
}  // namespace apollo
