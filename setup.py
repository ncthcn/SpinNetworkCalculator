from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "spin_backend",
        ["spin_backend/spin_backend.cpp"],
    ),
]

setup(
    name="spin_backend",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

