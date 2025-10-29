#include <bits/stdc++.h>
#include <memory>
#include "fun.hpp"
using namespace std;
using namespace gradlib;

int main() {

  auto f = Fun(10);
  auto g = Fun(3);
  auto z = g;
  auto h = f / g;

  h.backprop();
  cout << fixed << setprecision(5) << h.grad() << '\n';
  cout << fixed << setprecision(5) << f.grad() << '\n';
  cout << fixed << setprecision(5) << g.grad() << '\n';
  cout << fixed << setprecision(5) << z.grad() << '\n';
  h.destroy_graph_recursively();

}