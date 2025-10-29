#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <vector>

namespace gradlib {

struct _Fun;

struct Edge {
    std::shared_ptr<_Fun> ptr;
    std::function<double(double)> grad_fn;
};

struct _Fun {
    double _val = 0.0;
    double _grad = 0.0;
    std::vector<Edge> edges;

    _Fun(double val = 0.0) : _val(val) {}

    void backprop(double grad = 1.0) {
        _grad += grad;
        for (auto& e : edges) {
            e.ptr->backprop(e.grad_fn(grad));
        }
    }

    void destroy_graph_recursively(bool clear_mem) {
        _grad = 0.0;
        for (auto& e : edges) {
            e.ptr->destroy_graph_recursively(clear_mem);
        }
        edges.clear();
        if(clear_mem) {
            edges.shrink_to_fit();
        }
    }

    double grad() const { return _grad; }
    double value() const { return _val; }
};

struct Fun {
    std::shared_ptr<_Fun> node;

    // Constructors
    Fun() : node(std::make_shared<_Fun>()) {}
    Fun const& operator=(Fun const& other) { node = other.node; return *this; }
    Fun(Fun const& other) : node(other.node) {}
    Fun(Fun&& other) : node(move(other.node)) {}
    Fun(double val) : node(std::make_shared<_Fun>(val)) {}

    // Arithmetic operators
    Fun operator+(const Fun& other) const {
        Fun res(node->_val + other.node->_val);
        res.node->edges.push_back({node,       [](double g) { return g; }});
        res.node->edges.push_back({other.node, [](double g) { return g; }});
        return res;
    }

    Fun operator-(const Fun& other) const {
        Fun res(node->_val - other.node->_val);
        res.node->edges.push_back({node,       [](double g) { return g; }});
        res.node->edges.push_back({other.node, [](double g) { return -g; }});
        return res;
    }

    Fun operator*(const Fun& other) const {
        Fun res(node->_val * other.node->_val);
        res.node->edges.push_back({node,       [ov = other.node->_val](double g) { return g * ov; }});
        res.node->edges.push_back({other.node, [tv = node->_val](double g) { return g * tv; }});
        return res;
    }

    Fun operator/(const Fun& other) const {
        Fun res(node->_val / other.node->_val);
        res.node->edges.push_back({node,       [ov = other.node->_val](double g) { return g / ov; }});
        res.node->edges.push_back({other.node, [tv = node->_val, ov = other.node->_val](double g) { return g * tv * (-1.0 / (ov * ov)); }});
        return res;
    }

    // Accessors
    double value() const { return node->value(); }
    double grad() const { return node->grad(); }
    void non_grad_add(double val) const { node->_val += val; }

    // Backpropagation
    void backprop() { node->backprop(); }
    void destroy_graph_recursively(bool clear_mem = true) {
        node->destroy_graph_recursively(clear_mem);
    }

    explicit operator double() const { return value(); }
};

}