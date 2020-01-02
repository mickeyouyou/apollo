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

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

namespace apollo {
namespace planning {

bool InitialCuda();

inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    printf("CUDA Runtime Error: %s:%d, ", __FILE__, __LINE__);
    printf("Code: %d, message: %s\n", result, cudaGetErrorString(result));
  }
  return result;
}

void data_feed_util(int *iRow, int *jCol, unsigned int *rind_L,
                    unsigned int *cind_L, const int nnz_L);
void data_set_util(double *dst, const double *src, const int size);

template <typename T>
__global__ void kernel_objective(int n, const T *x, double ts_, int horizon_,
                                 double *last_time_u_, double *xWS_,
                                 double *xf_, int obstacles_num_,
                                 int obstacles_edges_sum_, T *obj_value);

template <class T>
void evalue_objective(int n, const T *x, double ts_, int horizon_,
                      double *last_time_u_, double *xWS_, double *xf_,
                      int obstacles_num_, int obstacles_edges_sum_,
                      T *obj_value);

}  // namespace planning
}  // namespace apollo
