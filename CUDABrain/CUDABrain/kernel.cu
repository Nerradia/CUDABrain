
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "cudaWorker.h"

int main()
{
	printf("Boop !  \n");

	cudaWorker* testWorker;

	testWorker = new cudaWorker(8);

	printf("Deleting...\n");

	delete testWorker;

	return 0;
};