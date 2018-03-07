#include "cudaWorker.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "parameters.h"

#define DBG(a, b) if(a) fprintf(stderr, "%s : %s, line %d, file %s.\n", #a, b, __LINE__, __FILENAME__)

cudaWorker::cudaWorker(int nNetworks)
{

	DBG(DBG_CUDAWORKER, "Entering cudaWorker()");

	if (nNetworks < 1 || nNetworks > 1024) {
		fprintf(stderr, "Bad number of network ! Requested : %d, expecting something between 1 and 1024.\n", nNetworks);
		exit(1);
	}

	this->nNetworks = nNetworks;

	networks = new cudanetwork[nNetworks];

	for (int i = 0; i < nNetworks; i++) {

		networks[i].interconnects = NULL;
		networks[i].memories = NULL;
		networks[i].constants = NULL;

		networks[i].inputs = NULL;
		networks[i].outputs = NULL;

		networks[i].dev_interconnects = NULL;
		networks[i].dev_memories = NULL;
		networks[i].dev_constants = NULL;

		networks[i].dev_inputs = NULL;
		networks[i].dev_lvl1 = NULL;
		networks[i].dev_lvl2 = NULL;
		networks[i].dev_lvl3 = NULL;
		networks[i].dev_outputs = NULL;
		networks[i].ready = 0;

		/* Simple malloc here, since we will copy it to GPU only once,
		and page-locked memory is said to be used only when really needed */
		networks[i].dev_interconnects = (float*)malloc(NETWORK_PARAMETERS_TOTAL_SIZE * sizeof(float));
		DBG(DBG_MALLOCS, "dev_interconnects");

		if (networks[i].dev_interconnects == NULL) {
			fprintf(stderr, "Malloc returned NULL pointer, network %d.\n", i);
			exit(1);
		}

		/* Placing pointers at the right area of allocated space */
		networks[i].memories  = networks[i].interconnects + NETWORK_INTERCONNECTS_SIZE;
		networks[i].constants = networks[i].memories      + NETWORK_MEMORIES_SIZE;
	}
	DBG(DBG_CUDAWORKER, "Leaving cudaWorker()");
}


cudaWorker::~cudaWorker()
{
	DBG(DBG_CUDAWORKER, "Entering ~cudaWorker()");
	/* Try to free what has been allocated on host and device */
	for (int i = 0; i < nNetworks; i++) {
		DBG(DBG_MALLOCS, "Freeing everything");
		safeFree(networks[i].interconnects);

		safeFreeCUDAHost(networks[i].inputs);
		safeFreeCUDAHost(networks[i].outputs);

		safeFreeCUDA(networks[i].dev_interconnects);
		safeFreeCUDA(networks[i].dev_inputs);
		safeFreeCUDA(networks[i].dev_lvl1);
		safeFreeCUDA(networks[i].dev_lvl2);
		safeFreeCUDA(networks[i].dev_lvl3);
		safeFreeCUDA(networks[i].dev_outputs);
	}


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}


	DBG(DBG_CUDAWORKER, "Leaving ~cudaWorker()");
}

void cudaWorker::safeFree(void* p) {
	if (p != NULL) free(p);
}
void cudaWorker::safeFreeCUDA(void* p) {
	if (p != NULL) cudaFree(p);
}
void cudaWorker::safeFreeCUDAHost(void* p) {
	if (p != NULL) cudaFreeHost(p);
}

int cudaWorker::initCuda() {
	
	/* First verify that every network has been loaded */
	for (int i = 0; i < nNetworks; i++) {
		if (networks[i].ready == 0) {
			fprintf(stderr, "Error, initCuda() called but network %d isn't ready.\n", i);
			return 1;
		}
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 1;
	}

	for (int i = 0; i < nNetworks; i++) {

		/* Allocation on device side of network parameters */
		cudaStatus = cudaMalloc((void**)&networks[i].dev_interconnects, NETWORK_PARAMETERS_TOTAL_SIZE * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for data of network %d.\n", i);
			return 1;
		}

		/* Copy data on device */
		cudaMemcpy((void*)networks[i].dev_interconnects, networks[i].interconnects, NETWORK_PARAMETERS_TOTAL_SIZE, cudaMemcpyHostToDevice);

		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed for data of network %d.\n", i);
			return 1;
		}

		/* Placing pointers at the right area of allocated space */
		networks[i].dev_memories  = networks[i].dev_interconnects + NETWORK_INTERCONNECTS_SIZE;
		networks[i].dev_constants = networks[i].dev_memories      + NETWORK_MEMORIES_SIZE;

		/* Allocation on device side of network level (in, internals and out) */
		cudaStatus = cudaMalloc((void**)&networks[i].dev_inputs, N_INPUTS * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for inputs of network %d.\n", i);
			return 1;
		}

		cudaStatus = cudaMalloc((void**)&networks[i].dev_lvl1, N_NLVL1 * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for level 1 of network %d.\n", i);
			return 1;
		}

		cudaStatus = cudaMalloc((void**)&networks[i].dev_lvl2, N_NLVL2 * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for layer 2 of network %d.\n", i);
			return 1;
		}

		cudaStatus = cudaMalloc((void**)&networks[i].dev_lvl3, N_NLVL3 * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for layer 3 of network %d.\n", i);
			return 1;
		}

		cudaStatus = cudaMalloc((void**)&networks[i].dev_outputs, N_OUTPUTS * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for output of network %d.\n", i);
			return 1;
		}

		/* Allocation of the host page-locked memory for network I/O */
		cudaStatus = cudaMallocHost((void**)&networks[i].inputs, N_INPUTS * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost failed for output of network %d.\n", i);
			return 1;
		}

		cudaStatus = cudaMallocHost((void**)&networks[i].outputs, N_OUTPUTS * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost failed for output of network %d.\n", i);
			return 1;
		}

	}
}

int cudaWorker::computeStep() {
	return 0;

}


int cudaWorker::setInterconnect(int id, int stage, int input, int output, float value) {
	if (id < 0 || id >= nNetworks) {
		fprintf(stderr, "Error in setInterconnect : expected an ID between 0 and %d, got %d\n.", nNetworks - 1, id);
		return 1;
	}

	if (stage < 0 || stage >= 4) {
		fprintf(stderr, "Error in setInterconnect : expected stage number between 0 and 3, got %d\n.", stage);
		return 1;
	}

	if (input < 0) {
		fprintf(stderr, "Error in setInterconnect : input ID is %d, , expected something positive\n", input);
		return 1;
	}

	networks[id].interconnects[getInterconnectID(stage, input, output)] = value;

}

float cudaWorker::getInterconnect(int id, int stage, int input, int output) {
	return 0.;

}

void cudaWorker::setMemory(int id, int stage, int neuron, float value) {

}

float cudaWorker::getMemory(int id, int stage, int neuron) {
	return 0.;

}

void cudaWorker::setConstant(int id, int stage, int neuron, float value) {

}

float cudaWorker::getConstant(int id, int stage, int neuron) {
	return 0.;

}



void cudaWorker::setInputData(int id, float* values) {

}

void cudaWorker::getOutputData(int id, char* values) {


}

int cudaWorker::getInterconnectID(int stage, int input, int output) {
	int dest;
	int inMax;
	int outMax;

	switch (stage) {

	case 0: // Input to first layer

		dest = N_NLVL1 * input + output;

		inMax = N_INPUTS;
		outMax = N_NLVL1;
		break;

	case 1: // Layer 1 to layer 2
		dest = N_INPUTS * N_NLVL1
			+ N_NLVL2 * input + output;

		inMax = N_NLVL1;
		outMax = N_NLVL2;
		break;

	case 2: // Layer 2 to 3
		dest = N_INPUTS * N_NLVL1
			+ N_NLVL1 * N_NLVL2
			+ N_NLVL2 * input + output;

		inMax = N_NLVL2;
		outMax = N_NLVL3;

	case 3: // Layer 3 to output
		dest = N_INPUTS * N_NLVL1
			+ N_NLVL1 * N_NLVL2
			+ N_NLVL2 * N_NLVL3
			+ N_NLVL3 * input + output;

		inMax = N_NLVL3;
		outMax = N_OUTPUTS;
	}

	if (input >= inMax) {
		fprintf(stderr, "Error : input ID is %d, expected something between 0 and %d\n", input, inMax);
		exit(1);
	}

	if (output >= outMax) {
		fprintf(stderr, "Error : output ID is %d, expected something between 0 and %d\n", output, outMax);
		exit(1);
	}

	return dest;
}