from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="ternlogic_cuda",
    ext_modules=[
        CUDAExtension(
            name="ternlogic_cuda",
            sources=[
                "ternlogic.cpp",
                "ternlogic_kernel.cu"
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)