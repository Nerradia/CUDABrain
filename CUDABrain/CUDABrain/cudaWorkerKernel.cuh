#pragma once

#include "cudaWorker.h"

void cudaWorkerKernellCall(int nblocks, int nthreads, void* index, int step);


