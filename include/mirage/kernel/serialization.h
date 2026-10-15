#pragma once
#include "mirage/kernel/graph.h"

namespace mirage::kernel {
void cython_to_json(Graph const *graph, char const *filename);
Graph *cython_from_json(char const *filename);
}
