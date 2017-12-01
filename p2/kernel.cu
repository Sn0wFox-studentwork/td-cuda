#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../wb.h"

#include <stdio.h>


#define BLOCK_SIZE 256
#define STENCIL_POINTS 7
#define value(tab, i, j, k) tab[(i * width + j) * depth + k]
#define in(i, j, k) value(input, i, j, k)
#define out(i, j, k) value(output, i, j, k)
#define MIN(v1, v2) (v1 < v2 ? v1 : v2)
#define MAX(v1, v2) (v1 > v2 ? v1 : v2)
#define CLAMP(val, start, end) MAX(MIN(val, end), start)
#define wbCheck(stmt) \
do { \
  cudaError_t err = stmt; \
  if (err != cudaSuccess) { \
  wbLog(ERROR, "Failed to run stmt ", #stmt); \
  wbLog(ERROR, "Got CUDA error ... ", cudaGetErrorString(err)); \
  return -1; \
  } \
} while (0)


__global__ void stencil(float *output, float *input, int width, int height, int depth) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;

  for (int k = 1; k < depth - 2; k++) {
    int res = in(i, j, k + 1) + in(i, j, k - 1) + in(i, j + 1, k) + in(i, j - 1, k) + in(i + 1, j, k) + in(i - 1, j, k) - 6 * in(i, j, k);
    out(i, j, k) = CLAMP(res, 0, 255);
  }
}


static void launch_stencil(float *deviceOutputData, float *deviceInputData, int width, int height, int depth) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((width - 1) / block.x +1, (height -1)/block.y +1);
  stencil <<< block, grid >>> (deviceOutputData, deviceInputData, width, height, depth);
}


int main(int argc, char *argv[]) {
  wbArg_t arg;
  int width;
  int height;
  int depth;
  char *inputFile;
  wbImage_t input;
  wbImage_t output;
  float *hostInputData;
  float *hostOutputData;
  float *deviceInputData;
  float *deviceOutputData;

  arg = wbArg_read(argc, argv);

  inputFile = wbArg_getInputFile(arg, 0);
  input = wbImport(inputFile);
  width = wbImage_getWidth(input);
  height = wbImage_getHeight(input);
  depth = wbImage_getChannels(input);
  output = wbImage_new(width, height, depth);
  hostInputData = wbImage_getData(input);
  hostOutputData = wbImage_getData(output);

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputData, width * height * depth * sizeof(float));
  cudaMalloc((void **)&deviceOutputData, width * height * depth * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputData, hostInputData, width * height * depth * sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  launch_stencil(deviceOutputData, deviceInputData, width, height, depth);
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputData, deviceOutputData, width * height * depth * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  cudaFree(deviceInputData);
  cudaFree(deviceOutputData);
  wbImage_delete(output);
  wbImage_delete(input);
  return 0;
}

