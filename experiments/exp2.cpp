#include <bits/stdc++.h>
#include <memory>
#include "vec.hpp"
using namespace std;
using namespace gradlib;

int main() {

  Vec data(100'000);
  Vec parameters(data.size());

  default_random_engine eng(random_device{}());

  for(Fun& f : data.v) {
    f = uniform_real_distribution<double>(-100, 100)(eng);
  }

  for(Fun& f : parameters.v) {
    f = uniform_real_distribution<double>(-100, 100)(eng);
  }

  double alpha = 0.3;

  for(int ep = 0; ep < 20; ep++) {
    cerr << "epoch: " << ep << '\n';
    auto sub = data - parameters;
    Fun loss = (sub).dot(sub);
    cerr << "loss: " << loss.value() << '\n';
    loss.backprop();
    // cout << "grad[] = { "; for(Fun const& g : parameters.v) cout << g.grad() << ", "; cout << "}\n";
    // cout << "val[] = { "; for(Fun const& g : parameters.v) cout << g.value() << ", "; cout << "}\n";
    for(int i = 0; i < int(parameters.size()); i++) {
      parameters[i].non_grad_add(parameters[i].grad() * (-alpha));
    }
    loss.destroy_graph_recursively();
  }

}
