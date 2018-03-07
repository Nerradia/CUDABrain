#pragma once

void cudaWorkerKernellCall(int nblocks, int nthreads, float* networkData, float* inputs, int ninputs, float* ouputs, int noutputs);
