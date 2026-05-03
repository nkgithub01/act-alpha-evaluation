#include "solver.h"
#include "storage.h"

operations_research::Solver solver("sched_alloc");

std::map<std::string, Storage *> g_storage = {
    {"HBM", new Storage("HBM", {HBM_SIZE}, 0)},
	{"SPAD", new Storage("SPAD", {16384,16}, 0)},
	{"ACC", new Storage("ACC", {1024,16}, 0)}
};
