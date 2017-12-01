#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../wb.h"

#include <stdio.h>

#ifndef __CUDACC__
void atomicAdd(void *address, int rightSide);
void __syncthreads();
#endif


#define NUM_BINS 256
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define BLOCK_SIZE 256


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
        file, line);
        if (abort)
            exit(code);
    }
}


__global__
void histo(unsigned int *data, unsigned int *bins, int len) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  int i = id;
  int totalNumberOfThreads = blockDim.x * gridDim.x;

  // Init shared memory (faster than global)
  __shared__ unsigned int privateHisto[NUM_BINS];
  while (i < NUM_BINS) {
    privateHisto[i] = 0;
    i += totalNumberOfThreads;
  }
  __syncthreads();

  // Compute histo locally (in shared memory)
  i = id;
  while (i < len) {
    atomicAdd(&(privateHisto[data[i]]), 1);
    i += totalNumberOfThreads;
  }
  __syncthreads();

  // Copy histo in global memory
  i = id;
  while (i < NUM_BINS) {
    atomicAdd(&(bins[i]), privateHisto[i]);
    i += totalNumberOfThreads;
  }
}


void printHisto(unsigned int *histo, unsigned int len) {
  for (unsigned int i = 0; i < len; i++) {
    std::cout << "char " << (char)i << " : " << histo[i] << std::endl;
  }
}



int main(int argc, char *argv[]) {
    wbArg_t args;
    int inputLength;
    unsigned int *hostInput;
    unsigned int *hostBins;
    unsigned int *deviceInput;
    unsigned int *deviceBins;
    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0), &inputLength, "Integer");
    hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);
    wbLog(TRACE, "The number of bins is ", NUM_BINS);

    int size = inputLength * sizeof(unsigned int);
    int binSize = NUM_BINS * sizeof(unsigned int);

    wbTime_start(GPU, "Allocating GPU memory.");
    cudaMalloc((void **)&deviceInput, size);
    cudaMalloc((void **)&deviceBins, binSize);
    CUDA_CHECK(cudaDeviceSynchronize());
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    cudaMemcpy(deviceInput, hostInput, size, cudaMemcpyHostToDevice);
    cudaMemset(deviceBins, 0, binSize);
    CUDA_CHECK(cudaDeviceSynchronize());
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    // TODO: find good values
    dim3 block(BLOCK_SIZE);
    dim3 grid((BLOCK_SIZE - 1) / inputLength + 1);

    // Launch kernel
    // ----------------------------------------------------------
    wbLog(TRACE, "Launching kernel");
    wbTime_start(Compute, "Performing CUDA computation");
    histo <<< grid, block >>> (deviceInput, deviceBins, inputLength);
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    cudaMemcpy(hostBins, deviceBins, binSize, cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaDeviceSynchronize());
    wbTime_stop(Copy, "Copying output memory to the CPU");

    printHisto(hostBins, NUM_BINS);

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceBins);
    wbTime_stop(GPU, "Freeing GPU Memory");
 
    free(hostBins);
    free(hostInput);
    return 0;
}
