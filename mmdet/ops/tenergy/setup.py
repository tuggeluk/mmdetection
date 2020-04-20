from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

NVCC_ARGS = [
    '-D__CUDA_NO_HALF_OPERATORS__',
    '-D__CUDA_NO_HALF_CONVERSIONS__',
    '-D__CUDA_NO_HALF2_OPERATORS__',
]

setup(
    name='tenergy',
    ext_modules=[
        CUDAExtension(
            'tenergy_cuda',
            ['src/tenergy_cuda.cpp', 'src/tenergy_cuda_kernel.cu'],
            extra_compile_args={
                'cxx': [],
                'nvcc': NVCC_ARGS
            })
    ],
    cmdclass={'build_ext': BuildExtension})
