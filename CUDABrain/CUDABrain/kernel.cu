
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include "cudaWorker.h"

int main()
{
	printf("Boop \n");

	FILE* debugOutput = fopen("./out.txt", "w");
	if (debugOutput == NULL) {
		printf("Error opening debug file.\n");
		exit(1);
	}

	for (int a = 1; a <= 128; a*=2) {

		cudaWorker* testWorker;

		int nbnetworks = 8 * a;

		testWorker = new cudaWorker(nbnetworks);

		for (int i = 0; i < nbnetworks; i++) {
			//printf("Filling network interconnects %d with zeroes...\n", i);

			/* For each input */
			for (int j = 0; j < N_INPUTS; j++) {

				/* For each neuron of layer 1 */
				for (int k = 0; k < N_NLVL1; k++) {
					testWorker->setInterconnect(i, 0, j, k, 0.f);
				}

			}

			/* For each layer 1 */
			for (int j = 0; j < N_NLVL1; j++) {

				/* For each layer 2 */
				for (int k = 0; k < N_NLVL2; k++) {
					testWorker->setInterconnect(i, 1, j, k, 0.f);
				}

				testWorker->setMemory(i, 1, j, 0.f);
				testWorker->setConstant(i, 1, j, 0.f);
			}

			/* For each layer 2 */
			for (int j = 0; j < N_NLVL2; j++) {

				/* For each layer 3 */
				for (int k = 0; k < N_NLVL3; k++) {
					testWorker->setInterconnect(i, 2, j, k, 0.f);
				}

				testWorker->setMemory(i, 2, j, 0.f);
				testWorker->setConstant(i, 2, j, 0.f);
			}

			/* For each layer 3 */
			for (int j = 0; j < N_NLVL3; j++) {

				/* For each output */
				for (int k = 0; k < N_OUTPUTS; k++) {
					testWorker->setInterconnect(i, 3, j, k, 0.f);
				}

				testWorker->setMemory(i, 3, j, 0.f);
				testWorker->setConstant(i, 3, j, 0.f);
			}

			for (int j = 0; j < N_OUTPUTS; j++) {
				testWorker->setMemory(i, 4, j, 0.f);
				testWorker->setConstant(i, 4, j, 0.f);
			}

			// Write everything on GPU memory
			testWorker->updateNetwork(i);
		}

		testWorker->setInterconnect(0, 0, 0, 0, 1.f);
		testWorker->setInterconnect(0, 1, 0, 5, 1.f);
		testWorker->setInterconnect(0, 2, 5, 0, 1.f);
		testWorker->setInterconnect(0, 3, 0, 0, 1.f);
		testWorker->setMemory(0, 3, 0, 1.f);

		// Update network 0 on GPU memory
		testWorker->updateNetwork(0);

		/* Get the table to write inputs */

		float* inputs = testWorker->getInputDataBuffer(0);
		float* outputs = testWorker->getOutputDataBuffer(0);

		/* Put everything to 0 */
		for (int i = 0; i < 128; i++) inputs[i] = 0.f;

		/* Except first input */
		inputs[0] = 1.0f;

		//printf("Compute 1000 step\n");

		clock_t start = clock();

		for (int i = 0; i < 100; i++) testWorker->computeStep();

		clock_t end = clock();
		float seconds = (float)(end - start) / CLOCKS_PER_SEC;
		printf("%d %f\n", nbnetworks, seconds);

		/* Get outputs */
		for (int i = 0; i < 4; i++)
			fprintf(debugOutput, "Outputs %3d : %f\n", i, outputs[i]);

		//printf("Deleting...\n");

		delete testWorker;
	}

	return 0;
};