#include "mirage/kernel/serialization.h"
#include <fstream>
#include <memory>
#include <stdexcept>

namespace mirage::kernel {
void cython_to_json(Graph const *graph, char const *filename) {
  json j;
  to_json(j, *graph);
  std::ofstream output(filename);
  if (!output) throw std::runtime_error("Cannot open graph output file");
  output << j;
  if (!output) throw std::runtime_error("Cannot write graph output file");
}

Graph *cython_from_json(char const *filename) {
  std::ifstream input(filename);
  if (!input) throw std::runtime_error("Cannot open graph input file");
  json j;
  input >> j;
  auto graph = std::make_unique<Graph>(dim3(1, 1, 1), true);
  from_json(j, *graph);
  return graph.release();
}
}
