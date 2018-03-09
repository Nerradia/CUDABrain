#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "cudaWorker.h"
#include "cudaWorkerKernel.cuh"

typedef struct {
	float* dev_interconnects;
	float* dev_memories;
	float* dev_constants;

	float* dev_inputs;
	float* dev_lvl1;
	float* dev_lvl2;
	float* dev_lvl3;
	float* dev_outputs;
} cudaNetworkGPUAdresses_kernel;

__global__ void cudaWorkerKernell(void * index_, int step) {
	cudaNetworkGPUAdresses_kernel *index = (cudaNetworkGPUAdresses_kernel*)index_;
	float* networkInterconnects;
	float* networkConstants;
	float* networkMemories;
	float* inputs;
	float* outputs;
	int nInputs;
	int nOutputs;

	switch (step) {
	case 0: // Inputs to stage 1
		networkInterconnects = index[blockIdx.x].dev_interconnects;
		networkConstants = index[blockIdx.x].dev_constants;
		networkMemories = index[blockIdx.x].dev_memories;
		inputs = index[blockIdx.x].dev_inputs;
		outputs = index[blockIdx.x].dev_lvl1;
		nInputs = N_INPUTS;
		nOutputs = N_NLVL1;
		break;

	case 1: // Stage 1 to stage 2
		networkInterconnects = index[blockIdx.x].dev_interconnects + N_NLVL1 * N_INPUTS;
		networkConstants = index[blockIdx.x].dev_constants + N_NLVL1;
		networkMemories = index[blockIdx.x].dev_memories + N_NLVL1;
		inputs = index[blockIdx.x].dev_lvl1;
		outputs = index[blockIdx.x].dev_lvl2;
		nInputs = N_NLVL1;
		nOutputs = N_NLVL2;
		break;

	case 2: // Stage 2 to stage 3
		networkInterconnects = index[blockIdx.x].dev_interconnects + N_NLVL1 * N_INPUTS + N_NLVL2 * N_NLVL1;
		networkConstants = index[blockIdx.x].dev_constants + N_NLVL1 + N_NLVL2;
		networkMemories = index[blockIdx.x].dev_memories + N_NLVL1 + N_NLVL2;
		inputs = index[blockIdx.x].dev_lvl2;
		outputs = index[blockIdx.x].dev_lvl3;
		nInputs = N_NLVL2;
		nOutputs = N_NLVL3;
		break;

	case 3: // Stage 3 to output
		networkInterconnects = index[blockIdx.x].dev_interconnects + N_NLVL1 * N_INPUTS + N_NLVL2 * N_NLVL1 + N_NLVL3 * N_NLVL2;
		networkConstants = index[blockIdx.x].dev_constants + N_NLVL1 + N_NLVL2 + N_NLVL3;
		networkMemories = index[blockIdx.x].dev_memories + N_NLVL1 + N_NLVL2 + N_NLVL3;
		inputs = index[blockIdx.x].dev_lvl3;
		outputs = index[blockIdx.x].dev_outputs;
		nInputs = N_NLVL3;
		nOutputs = N_OUTPUTS;
		break;

	default :
		printf("Error in kernel, received incorrect stage id.\n");
		return;
	}

	int threadID = threadIdx.x;
	int io_ratio = nInputs / nOutputs;



	/* Shared memory */
	__shared__ float cached_inputs[MAX_NLVL];

	float sum;

	/* Caching inputs in shared memory, because every thread will need it */
	for (int i = 0; i < io_ratio; i++) {
		cached_inputs[io_ratio * i + threadID] = inputs[io_ratio * i + threadID];
	}

	__syncthreads();

	/* Doing old_output * memCoefficient + constant in a single operation
	Global memory accesses are coalesced */
	sum = fmaf(outputs[threadID], networkMemories[threadID], networkConstants[threadID]);


	__syncthreads();

	/* Do some computations, global memory accesses are coalesced  */
	for (int i = 0; i < nInputs; i++) {
		sum = fmaf(cached_inputs[i], networkInterconnects[nOutputs * i + threadID], sum);
	}

	/* Write everything (coalesced write of 1 float per thread)*/
	outputs[threadID] = sum;

	/*if (blockIdx.x == 0)
		printf("Kernel thread %d, step %d, output %f\n", threadID, step, sum);*/
}


void cudaWorkerKernellCall(int nblocks, int nthreads, void* index, int step) {

	cudaWorkerKernell<<<nblocks, nthreads>>>(index, step);

	cudaError cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
		return;
	}
}
