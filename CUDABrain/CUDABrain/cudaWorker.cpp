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
		fprintf(stderr, "Bad number of networks ! Requested : %d, expecting something between 1 and 1024.\n", nNetworks);
		exit(1);
	}

	this->nNetworks = nNetworks;

	networks = new cudanetwork[nNetworks];
	networksGPU = new cudaNetworkGPUAdresses[nNetworks];

	for (int i = 0; i < nNetworks; i++) {

		networks[i].interconnects = NULL;
		networks[i].memories = NULL;
		networks[i].constants = NULL;

		networks[i].inputs = NULL;
		networks[i].outputs = NULL;

		networksGPU[i].dev_interconnects = NULL;
		networksGPU[i].dev_memories = NULL;
		networksGPU[i].dev_constants = NULL;

		networksGPU[i].dev_inputs = NULL;
		networksGPU[i].dev_lvl1 = NULL;
		networksGPU[i].dev_lvl2 = NULL;
		networksGPU[i].dev_lvl3 = NULL;
		networksGPU[i].dev_outputs = NULL;

		/* Simple malloc here, since we will copy it to GPU only once,
		and page-locked memory is said to be used only when really needed */
		networks[i].interconnects = (float*)malloc(NETWORK_PARAMETERS_TOTAL_SIZE * sizeof(float));
		DBG(DBG_MALLOCS, "dev_interconnects");

		if (networks[i].interconnects == NULL) {
			fprintf(stderr, "Malloc returned NULL pointer, network %d.\n", i);
			exit(1);
		}

		/* Placing pointers at the right area of allocated space */
		networks[i].memories  = networks[i].interconnects + NETWORK_INTERCONNECTS_SIZE;
		networks[i].constants = networks[i].memories      + NETWORK_MEMORIES_SIZE;
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		exit(1);
	}

	/* Allocation on device side for table of addresses */
	cudaStatus = cudaMalloc((void**)&dev_networksGPU, nNetworks * sizeof(cudaNetworkGPUAdresses));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for device address indexes.\n");
		exit(1);
	}

	for (int i = 0; i < nNetworks; i++) {

		/* Allocation on device side of network parameters */
		cudaStatus = cudaMalloc((void**)&networksGPU[i].dev_interconnects, NETWORK_PARAMETERS_TOTAL_SIZE * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for data of network %d.\n", i);
			exit(1);
		}

		/* Placing pointers at the right area of allocated space */
		networksGPU[i].dev_memories = networksGPU[i].dev_interconnects + NETWORK_INTERCONNECTS_SIZE;
		networksGPU[i].dev_constants = networksGPU[i].dev_memories + NETWORK_MEMORIES_SIZE;

		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed for data of network %d.\n", i);
			exit(1);
		}

		/* Allocation on device side of network level (in, internals and out) */
		cudaStatus = cudaMalloc((void**)&networksGPU[i].dev_inputs, N_INPUTS * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for inputs of network %d.\n", i);
			exit(1);
		}

		cudaStatus = cudaMalloc((void**)&networksGPU[i].dev_lvl1, N_NLVL1 * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for level 1 of network %d.\n", i);
			exit(1);
		}

		cudaStatus = cudaMalloc((void**)&networksGPU[i].dev_lvl2, N_NLVL2 * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for layer 2 of network %d.\n", i);
			exit(1);
		}

		cudaStatus = cudaMalloc((void**)&networksGPU[i].dev_lvl3, N_NLVL3 * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for layer 3 of network %d.\n", i);
			exit(1);;
		}

		cudaStatus = cudaMalloc((void**)&networksGPU[i].dev_outputs, N_OUTPUTS * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for output of network %d.\n", i);
			exit(1);
		}

		/* Allocation of the host page-locked memory for network I/O */
		cudaStatus = cudaMallocHost((void**)&networks[i].inputs, N_INPUTS * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost failed for output of network %d.\n", i);
			exit(1);
		}

		cudaStatus = cudaMallocHost((void**)&networks[i].outputs, N_OUTPUTS * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost failed for output of network %d.\n", i);
			exit(1);
		}
	}

	/* Send pointer table to GPU */
	cudaMemcpy((void*)dev_networksGPU, networksGPU, nNetworks * sizeof(cudaNetworkGPUAdresses), cudaMemcpyHostToDevice);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed for table data.\n");
		exit(1);
	}


	DBG(DBG_CUDAWORKER, "Leaving cudaWorker()");
}


cudaWorker::~cudaWorker()
{
	DBG(DBG_CUDAWORKER, "Entering ~cudaWorker()");
	/* Try to free what has been allocated on host and device */
	for (int i = 0; i < nNetworks; i++) {
		DBG(DBG_MALLOCS, "Freeing everything");

		DBG(DBG_MALLOCS, "networks interconnects");
		safeFree(networks[i].interconnects);
		networks[i].interconnects = NULL;

		DBG(DBG_MALLOCS, "networks inputs");
		safeFreeCUDAHost(networks[i].inputs);
		networks[i].inputs = NULL;

		DBG(DBG_MALLOCS, "networks outputs");
		safeFreeCUDAHost(networks[i].outputs);
		networks[i].outputs = NULL;

		DBG(DBG_MALLOCS, "device memory");
		safeFreeCUDA(networksGPU[i].dev_interconnects);
		networksGPU[i].dev_interconnects = NULL;

		safeFreeCUDA(networksGPU[i].dev_inputs);
		networksGPU[i].dev_inputs = NULL;
		
		safeFreeCUDA(networksGPU[i].dev_lvl1);
		networksGPU[i].dev_lvl1 = NULL;

		safeFreeCUDA(networksGPU[i].dev_lvl2);
		networksGPU[i].dev_lvl2 = NULL;

		safeFreeCUDA(networksGPU[i].dev_lvl3);
		networksGPU[i].dev_lvl3 = NULL;

		safeFreeCUDA(networksGPU[i].dev_outputs);
		networksGPU[i].dev_outputs = NULL;
	}


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}


	DBG(DBG_CUDAWORKER, "Leaving ~cudaWorker()");
}

void cudaWorker::safeFree(float* p) {
	if (p != NULL) free(p);
}
void cudaWorker::safeFreeCUDA(float* p) {
	if (p != NULL) cudaFree(p);
}
void cudaWorker::safeFreeCUDAHost(float* p) {
	if (p != NULL) cudaFreeHost(p);
}

void cudaWorker::updateNetwork(int id) {

	/* Copy data on device */
	cudaMemcpy((void*)networksGPU[id].dev_interconnects, networks[id].interconnects, NETWORK_PARAMETERS_TOTAL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

}

int cudaWorker::computeStep() {

	/* Copy inputs on device */
	for (int i = 0; i < nNetworks; i++) {
		cudaMemcpy((void*)networksGPU[i].dev_inputs, networks[i].inputs, N_INPUTS * sizeof(float), cudaMemcpyHostToDevice);

		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed for input data of network %d.\n", i);
			return 1;
		}
	}

	/* Call the Kernel for each stage*/

	cudaWorkerKernellCall(nNetworks, N_NLVL1, dev_networksGPU, 0);
	cudaWorkerKernellCall(nNetworks, N_NLVL2, dev_networksGPU, 1);
	cudaWorkerKernellCall(nNetworks, N_NLVL3, dev_networksGPU, 2);
	cudaWorkerKernellCall(nNetworks, N_OUTPUTS, dev_networksGPU, 3);

	/* Copy outputs from device */
	for (int i = 0; i < nNetworks; i++) {
		cudaMemcpy(networks[i].outputs, networksGPU[i].dev_outputs, N_OUTPUTS * sizeof(float), cudaMemcpyDeviceToHost);

		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed for input data of network %d.\n", i);
			return 1;
		}
	}

	return 0;

}


int cudaWorker::setInterconnect(int id, int stage, int input, int output, float value) {
	
	if (id < 0 || id >= nNetworks) {
		fprintf(stderr, "Error in setInterconnect : expected an ID between 0 and %d, got %d.\n", nNetworks - 1, id);
		exit(1);
	}

	if (stage < 0 || stage >= 4) {
		fprintf(stderr, "Error in setInterconnect : expected stage number between 0 and 3, got %d.\n", stage);
		exit(1);
	}

	if (input < 0) {
		fprintf(stderr, "Error in setInterconnect : input ID is %d, , expected something positive.\n", input);
		exit(1);
	}
	
	networks[id].interconnects[getInterconnectID(stage, input, output)] = value;

	return 0;
}

float cudaWorker::getInterconnect(int id, int stage, int input, int output) {
	if (id < 0 || id >= nNetworks) {
		fprintf(stderr, "Error in getInterconnect : expected an ID between 0 and %d, got %d\n.", nNetworks - 1, id);
		exit(1);
	}

	if (stage < 0 || stage >= 3) {
		fprintf(stderr, "Error in getInterconnect : expected stage number between 0 and 3, got %d\n.", stage);
		exit(1);
	}

	if (input < 0) {
		fprintf(stderr, "Error in getInterconnect : input ID is %d, , expected something positive\n", input);
		exit(1);
	}

	return networks[id].interconnects[getInterconnectID(stage, input, output)];

}

void cudaWorker::setMemory(int id, int stage, int neuron, float value) {

	if (id < 0 || id >= nNetworks) {
		fprintf(stderr, "Error in setMemory : expected an ID between 0 and %d, got %d\n.", nNetworks - 1, id);
		exit(1);
	}

	if (stage < 1 || stage > 4) {
		fprintf(stderr, "Error in setMemory : expected stage number between 0 and 3, got %d\n.", stage);
		exit(1);
	}

	if (neuron < 0) {
		fprintf(stderr, "Error in setMemory : neuron ID is %d, expected something positive\n", neuron);
		exit(1);
	}

	networks[id].memories[getConstOrMemID(stage, neuron)] = value;
}

float cudaWorker::getMemory(int id, int stage, int neuron) {

	if (id < 0 || id >= nNetworks) {
		fprintf(stderr, "Error in setMemory : expected an ID between 0 and %d, got %d\n.", nNetworks - 1, id);
		exit(1);
	}

	if (stage < 1 || stage > 4) {
		fprintf(stderr, "Error in setMemory : expected stage number between 0 and 3, got %d\n.", stage);
		exit(1);
	}

	if (neuron < 0) {
		fprintf(stderr, "Error in setMemory : neuron ID is %d, expected something positive\n", neuron);
		exit(1);
	}

	return networks[id].memories[getConstOrMemID(stage, neuron)];

}

void cudaWorker::setConstant(int id, int stage, int neuron, float value) {
	if (id < 0 || id >= nNetworks) {
		fprintf(stderr, "Error in setMemory : expected an ID between 0 and %d, got %d\n.", nNetworks - 1, id);
		exit(1);
	}

	if (stage < 1 || stage > 4) {
		fprintf(stderr, "Error in setMemory : expected stage number between 0 and 3, got %d\n.", stage);
		exit(1);
	}

	if (neuron < 0) {
		fprintf(stderr, "Error in setMemory : neuron ID is %d, expected something positive\n", neuron);
		exit(1);
	}

	networks[id].constants[getConstOrMemID(stage, neuron)] = value;
}

float cudaWorker::getConstant(int id, int stage, int neuron) {

	if (id < 0 || id >= nNetworks) {
		fprintf(stderr, "Error in setMemory : expected an ID between 0 and %d, got %d\n.", nNetworks - 1, id);
		exit(1);
	}

	if (stage < 1 || stage > 4) {
		fprintf(stderr, "Error in setMemory : expected stage number between 0 and 3, got %d\n.", stage);
		exit(1);
	}

	if (neuron < 0) {
		fprintf(stderr, "Error in setMemory : neuron ID is %d, expected something positive\n", neuron);
		exit(1);
	}

	return networks[id].constants[getConstOrMemID(stage, neuron)];
}



float* cudaWorker::getInputDataBuffer(int id) {

	if (id < 0 || id >= nNetworks) {
		fprintf(stderr, "Error in setMemory : expected an ID between 0 and %d, got %d\n.", nNetworks - 1, id);
		exit(1);
	}

	return networks[id].inputs;
}

float* cudaWorker::getOutputDataBuffer(int id) {

	if (id < 0 || id >= nNetworks) {
		fprintf(stderr, "Error in setMemory : expected an ID between 0 and %d, got %d\n.", nNetworks - 1, id);
		exit(1);
	}

	return networks[id].outputs;
}

int cudaWorker::getConstOrMemID(int stage, int neuron) {

	int dest;
	int neuronMax;

	switch (stage) {

	case 1: // First layer

		dest = neuron;
		neuronMax = N_NLVL1;
		break;

	case 2: // Layer 2

		dest = N_NLVL1 + neuron;
		neuronMax = N_NLVL2;
		break;


	case 3: // Layer 3
		dest = N_NLVL1 + N_NLVL2 + neuron;
		neuronMax = N_NLVL3;
		break;

	case 4: // Output layer
		dest = N_NLVL1 + N_NLVL2 + N_NLVL3 + neuron;
		neuronMax = N_OUTPUTS;
		break;

	default:
		fprintf(stderr, "Stage not found at line %d\n", __LINE__);
		exit(1);
	}

	if (neuron >= neuronMax) {
		fprintf(stderr, "Error : neuron ID is %d, expected something between 0 and %d for layer %d\n", neuron, neuronMax, stage);
		exit(1);
	}

	return dest;
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
		dest = N_NLVL1 * N_INPUTS
			+ N_NLVL2 * input + output;

		inMax = N_NLVL1;
		outMax = N_NLVL2;
		break;

	case 2: // Layer 2 to 3
		dest = N_NLVL1 * N_INPUTS
			+ N_NLVL2 * N_NLVL1
			+ N_NLVL3 * input + output;

		inMax = N_NLVL2;
		outMax = N_NLVL3;
		break;

	case 3: // Layer 3 to output
		dest = N_NLVL1 * N_INPUTS
			+ N_NLVL2 * N_NLVL1
			+ N_NLVL3 * N_NLVL2
			+ N_OUTPUTS * input + output;

		inMax = N_NLVL3;
		outMax = N_OUTPUTS;
		break;

	default :
		fprintf(stderr, "Stage not found at line %d\n", __LINE__);
		exit(1);
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

