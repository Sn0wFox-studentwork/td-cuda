
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include "wb.h"


#define wbCheck(stmt) \
do { \
	cudaError_t err = stmt; \
	if (err != cudaSuccess) { \
		wbLog(ERROR, "Failed to run stmt ", #stmt); \
		wbLog(ERROR, "Got CUDA error ... ", cudaGetErrorString(err)); \
		return -1;\
	}\
} while (0)


const int BLOCK_SIZE = 256;


__global__
void vecAdd(float *v1, float *v2, float *vout, int len) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len) {
		vout[i] = v1[i] + v2[i];
	}
}

int main(int argc, char **argv) {
	wbArg_t args;
	int inputLength;
	float *hostInput1;
	float *hostInput2;
	float *hostOutput;
	float *deviceInput1;
	float *deviceInput2;
	float *deviceOutput;
	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
	hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
	hostOutput = (float *)malloc(inputLength * sizeof(float));
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The input length is ", inputLength);
	int size = inputLength * sizeof(float);

	wbTime_start(GPU, "Allocating GPU memory.");
	cudaMalloc((void **)&deviceInput1, size);
	cudaMalloc((void **)&deviceInput2, size);
	cudaMalloc((void **)&deviceOutput, size);
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceOutput, hostOutput, size, cudaMemcpyHostToDevice);
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	dim3 block(BLOCK_SIZE);
	dim3 grid((BLOCK_SIZE -1)/inputLength +1);

	wbTime_start(Compute, "Performing CUDA computation");
	vecAdd <<<grid, block>>> (deviceInput1, deviceInput2, deviceOutput, inputLength);
	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	cudaFree(deviceInput1);
	cudaFree(deviceInput2);
	cudaFree(deviceOutput);
	wbTime_stop(GPU, "Freeing GPU Memory");

	wbSolution(args, hostOutput, inputLength);
	free(hostInput1);
	free(hostInput2);
	free(hostOutput);

	return 0;
}
