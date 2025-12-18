#include <pybind11/pybind11.h>
#include <string>
#include <iostream>

// Temporary placeholder; we'll later call WIGXJPF etc.
double evaluate_graph(const std::string& graphml_path) {
    std::cout << "Evaluating graph: " << graphml_path << std::endl;
    // TODO: parse graphml, compute sums of 6j-symbols
    return 42.0;  // placeholder number
}

namespace py = pybind11;

PYBIND11_MODULE(spin_backend, m) {
    m.doc() = "C++ backend for 6j-symbol graph evaluation";
    m.def("evaluate_graph", &evaluate_graph,
          py::arg("graphml_path"),
          "Compute the numeric value associated with a graph.");
}

