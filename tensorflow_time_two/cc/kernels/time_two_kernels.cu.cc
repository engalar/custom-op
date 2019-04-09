/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "time_two.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;


//block-thread 3D-3D
/**
template <typename T>
__global__ void OptFlowCudaKernel(const T* in_1, const T* in_2, T* out)
{
    int threadId_3d_pre =  threadIdx.z*blockDim.x*blockDim.y;
    int threadId_3D_nxt = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    int blockId_3D_pre = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    int blockId_3D_nxt = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    int i_pre = threadId_3d_pre + (blockDim.x*blockDim.y*blockDim.z)*blockId_3D;
    int i_nxt = threadId_3D + (blockDim.x*blockDim.y*blockDim.z)*blockId_3D;
    
    feature_pre = ldg(in_1 + i_pre)
    feature_nxt = ldg(in_1 + i_nxt)
    (feature_pre - feature_nxt) * (feature_pre - feature_nxt)
    __shared__ float support[];
    support[threadIdx.x,threadIdx.y,threadIdx.z] =  (feature_pre - feature_nxt) * (feature_pre - feature_nxt);

    __syncthreads();
    if (threadIdx.z == 0) {
       // sum across feature channel
       
    }
}
**/

// Define the CUDA kernel.
template <typename T>
__global__ void TimeTwoCudaKernel(const int size, const T* in, T* out) {
  printf("gridDim.x   %d  ", gridDim.x);
  printf("blockIdx.x  %d  ", blockIdx.x);
  printf("threadIdx.x %d  ", threadIdx.x);
  printf("****************\n");
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    out[i] = 2 * ldg(in + i);
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct TimeTwoFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int size, const T* in, T* out) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    int block_count = 2;
    int thread_per_block = 2;
    TimeTwoCudaKernel<T>
        <<<block_count, thread_per_block, 0, d.stream()>>>(size, in, out);
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct TimeTwoFunctor<GPUDevice, float>;
template struct TimeTwoFunctor<GPUDevice, int32>;
}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
