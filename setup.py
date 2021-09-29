from torch.utils.cpp_extension import CUDAExtension
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension

ext_modules = [
    CUDAExtension('torch_operator_extension', [ 'torch_operator_extension.cpp', 'custom_operator.cu']),
]

setup(
    name='custom_operator',
    version='1.0',
    author="Pawel Kazmierczyk",
    author_email="pkazmier@amd.com",
    description="Custom operator template",
    install_requires=['torch'],
    # py_modules=["torch_custom_operator/custom_operator"],
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
