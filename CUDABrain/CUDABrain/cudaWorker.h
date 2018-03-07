#pragma once

#include "cudaWorkerKernel.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N_INPUTS 128
#define N_NLVL1 128
#define N_NLVL2 64
#define N_NLVL3 32
#define N_OUTPUTS 4

#define NETWORK_INTERCONNECTS_SIZE (N_INPUTS * N_NLVL1 + N_NLVL1 * N_NLVL2 + N_NLVL2 * N_NLVL3 + N_NLVL3 * N_OUTPUTS)
#define NETWORK_MEMORIES_SIZE (N_NLVL1 + N_NLVL2 + N_NLVL3 + N_OUTPUTS)
#define NETWORK_CONSTANTS_SIZE (N_NLVL1 + N_NLVL2 + N_NLVL3 + N_OUTPUTS)
#define NETWORK_PARAMETERS_TOTAL_SIZE (NETWORK_INTERCONNECTS_SIZE + NETWORK_MEMORIES_SIZE + NETWORK_CONSTANTS_SIZE)

typedef struct
{
	int ready;

	float* interconnects;
	float* memories;
	float* constants;

	float* inputs;
	float* outputs;
		
	float* dev_interconnects;
	float* dev_memories;
	float* dev_constants;

	float* dev_inputs;
	float* dev_lvl1;
	float* dev_lvl2;
	float* dev_lvl3;
	float* dev_outputs;
} cudanetwork;

class cudaWorker
{
public:

	cudaWorker(int nNetworks);

	/* Initialises GPU, allocate memory and copy network data */
	int initCuda();

	/* Networks configuration */
	/* Interconnection's value, s : stage, input neuron, output neuron, value */
	int setInterconnect(int id, int stage, int input, int output, float value);
	float getInterconnect(int id, int stage, int input, int output);

	/* Memory coefficient of a neuron */
	void setMemory(int id, int stage, int neuron, float value);
	float getMemory(int id, int stage, int neuron);

	/* Constant of a neuron */
	void setConstant(int id, int stage, int neuron, float value);
	float getConstant(int id, int stage, int neuron);


	/* Things to call at every simulation step */
	int computeStep();
	void setInputData(int id, float* values);
	void getOutputData(int id, char* values);


	~cudaWorker();

private:

	void safeFreeCUDA(void* p);
	void safeFreeCUDAHost(void *p);
	void safeFree(void* p);
	int getInterconnectID(int stage, int input, int output);
	
	cudanetwork* networks;
	int nNetworks;
	cudaError_t cudaStatus; 
};

