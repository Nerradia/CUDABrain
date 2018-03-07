#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void cudaWorkerKernell(float* networkData, float* inputs, int ninputs, float* ouputs, int noutputs) {



}


void cudaWorkerKernellCall(int nblocks, int nthreads, float* networkData, float* inputs, int ninputs, float* ouputs, int noutputs) {

//	cudaWorkerKenell << <nblocks, nthreads >> > (networkData, inputs, ninputs, outputs, noutputs);
}
