/*
 * Minimal CUDA Runtime API Stub
 * Compile: gcc -shared -fPIC -o libcudart.so.11.0 libcudart_stub.c
 */

#include <stdlib.h>
#include <string.h>

typedef int cudaError_t;
#define cudaSuccess 0

typedef struct cudaDeviceProp {
  char name[256];
  size_t totalGlobalMem;
  int major, minor, multiProcessorCount, clockRate;
  int deviceOverlap, kernelExecTimeoutEnabled, integrated;
  int canMapHostMemory, computeMode, concurrentKernels;
  int ECCEnabled, pciBusID, pciDeviceID, tccDriver;
} cudaDeviceProp;

cudaError_t cudaGetDeviceCount(int *count) {
  if (count)
    *count = 0;
  return cudaSuccess;
}
cudaError_t cudaSetDevice(int device) { return cudaSuccess; }
cudaError_t cudaGetDevice(int *device) {
  if (device)
    *device = 0;
  return cudaSuccess;
}
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
  memset(prop, 0, sizeof(cudaDeviceProp));
  strcpy(prop->name, "NIM-OV-Bridge CUDA Stub");
  prop->totalGlobalMem = 0;
  return cudaSuccess;
}
cudaError_t cudaMalloc(void **devPtr, size_t size) { return cudaSuccess; }
cudaError_t cudaFree(void *devPtr) { return cudaSuccess; }
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, int kind) {
  memcpy(dst, src, count);
  return cudaSuccess;
}
cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
  memset(devPtr, value, count);
  return cudaSuccess;
}
cudaError_t cudaDeviceSynchronize(void) { return cudaSuccess; }
cudaError_t cudaStreamCreate(void **stream) {
  if (stream)
    *stream = NULL;
  return cudaSuccess;
}
cudaError_t cudaStreamDestroy(void *stream) { return cudaSuccess; }
cudaError_t cudaStreamSynchronize(void *stream) { return cudaSuccess; }
const char *cudaGetErrorString(cudaError_t error) {
  return "cudaSuccess (stub)";
}
cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
  if (runtimeVersion)
    *runtimeVersion = 11080;
  return cudaSuccess;
}
cudaError_t cudaDriverGetVersion(int *driverVersion) {
  if (driverVersion)
    *driverVersion = 11080;
  return cudaSuccess;
}
cudaError_t cudaGetLastError(void) { return cudaSuccess; }
cudaError_t cudaPeekAtLastError(void) { return cudaSuccess; }