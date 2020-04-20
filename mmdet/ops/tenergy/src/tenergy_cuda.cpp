#include <ATen/ATen.h>
#include <torch/torch.h>

#include <cmath>
#include <vector>

int TENERGYLaucher(const at::Tensor masks, const int batch_size,
                        const int scale_factor,const int max_energy,
                        const int height, const int width, const int channels,
                        at::Tensor output);

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int tenergy_cuda(at::Tensor masks, int scale_factor, int max_energy,
                              at::Tensor output) {

  CHECK_INPUT(masks);
  CHECK_INPUT(output);
  at::DeviceGuard guard(masks.device());

  int batch_size = output.size(0);
  int num_channels = output.size(1);
  int data_height = output.size(2);
  int data_width = output.size(3);

  TENERGYLaucher(masks,batch_size, scale_factor,max_energy,
                        data_height, data_width, num_channels,output);

  return 1;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vote", &tenergy_cuda, "tenery_computation (CUDA)");
  
}
