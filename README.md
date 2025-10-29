# Automatic Differentiation Engine

A minimal, scalar-based reverse-mode automatic differentiation engine implemented in C++.
Builds a dynamic computational graph to perform backpropagation and compute gradients.

The goal of the project was to gain better understanding of the chain rule foundations of neural networks.

`experiments/exp2.cpp` demonstrates the engine by fitting a vector to a hidden random target vector.

# Future improvements

This project shows how closely related multivariate and single variable calculus are.

The next step is to implement the engine with Vec as a fundamental type.
This vectorized approach would drastically reduce the number of graph nodes and metadata overhead, making backpropagation for larger models, such as small MLPs, feasible.