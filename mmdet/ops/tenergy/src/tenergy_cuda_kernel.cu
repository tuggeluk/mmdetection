#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

using namespace at;  // temporal fix for pytorch<=0.4.1 (see #9848)

#define THREADS_PER_BLOCK 1024

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65536;
  return min(optimal_block_num, max_block_num);
}

__device__ inline int Loc2IndexSimple(const int h,const int w, 
                                const int height, const int width) {
  int index = h * width + w;
  return index;
}

__device__ inline int Loc2Index(const int n, const int c, const int h,
                                const int w, const int channel_num,
                                const int height, const int width) {
  int index = w + (h + (c + n * channel_num) * height) * width;
  return index;
}

template <typename scalar_t>
__global__ void TENERGY(const int nthreads,
                        const scalar_t *bottom_masks,
                        const int scale_factor, const int k,
                        const int height, const int width,const int channels,
                        scalar_t *top_data) {
  
    //int index = blockIdx.x * blockDim.x + threadIdx.x;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int pw = index % width;
    int ph = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    int mask_index_a =
              Loc2Index(n,c, ph, pw, channels, height, width);
    int mask_index_n1 =
              Loc2Index(n,c, ph + 1, pw,channels,height, width);
    int mask_index_n2 =
          Loc2Index(n,c, ph - 1, pw,channels,height, width);
    int mask_index_n3 =
          Loc2Index(n,c, ph, pw + 1,channels,height, width);
    int mask_index_n4 =
          Loc2Index(n,c, ph, pw - 1,channels,height, width);
    int mask_index_n5 =
          Loc2Index(n,c, ph + 1, pw + 1,channels,height, width);
    int mask_index_n6 =
          Loc2Index(n,c, ph - 1, pw - 1,channels,height, width);
    int mask_index_n7 =
          Loc2Index(n,c, ph - 1, pw + 1,channels,height, width);
    int mask_index_n8 =
          Loc2Index(n,c, ph + 1, pw - 1,channels,height, width);

    if(k==0)
      top_data[mask_index_a] =  bottom_masks[mask_index_a];
    else if (pw > 0 && ph > 0 && pw < width -1 && ph < height -1)
    {

        if (top_data[mask_index_n1]>=k && top_data[mask_index_n2]>=k
          && top_data[mask_index_n3]>=k && top_data[mask_index_n4]>=k
          && top_data[mask_index_n5]>=k && top_data[mask_index_n6]>=k
          && top_data[mask_index_n7]>=k && top_data[mask_index_n8]>=k)
          {
            top_data[mask_index_a] = top_data[mask_index_a] + 1.0;
          }
    }
}
}

int TENERGYLauncher(const at::Tensor masks, const int batch_size,
                              const int scale_factor,const int max_energy,
                              const int height,const int width, const int channels,
                              at::Tensor output) {
  const int output_size = batch_size * channels * height * width;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      masks.type(), "TENERGYLauncherVote", ([&] {
        const scalar_t *bottom_masks = masks.data<scalar_t>();
        scalar_t *top_data = output.data<scalar_t>();
        for( int k = 0;k < max_energy; k++)
        {
        TENERGY<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(output_size,
                bottom_masks, scale_factor, k, height, width, channels, top_data);
                cudaDeviceSynchronize();
        }
      }));

  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  return 1;
}

