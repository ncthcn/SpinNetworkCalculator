cmake_minimum_required(VERSION 3.14)
project(spin_backend LANGUAGES CXX)

find_package(pybind11 REQUIRED)

pybind11_add_module(spin_backend spin_backend.cpp)

