/*
 * Minimal CUDA Driver API Stub
 * Compile: gcc -shared -fPIC -o libcuda.so.1 libcuda_stub.c
 */

#include <stdint.h>
#include <string.h>

#define CUDA_SUCCESS 0
#define CUDA_ERROR_NO_DEVICE 100
#define CUDA_ERROR_INVALID_DEVICE 101

typedef int CUresult;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
typedef void* CUdeviceptr;
typedef int CUdevice;
typedef unsigned int CUjit_option;

CUresult cuInit(unsigned int flags) { return CUDA_SUCCESS; }
CUresult cuDriverGetVersion(int *driverVersion) {
    if (driverVersion) *driverVersion = 12020;
    return CUDA_SUCCESS;
}
CUresult cuDeviceGetCount(int *count) {
    if (count) *count = 0;
    return CUDA_SUCCESS;
}
CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    return CUDA_ERROR_INVALID_DEVICE;
}
CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    if (len > 0 && name) name[0] = '\0';
    return CUDA_ERROR_INVALID_DEVICE;
}
CUresult cuDeviceGetAttribute(int *value, int attrib, CUdevice dev) {
    if (value) *value = 0;
    return CUDA_ERROR_INVALID_DEVICE;
}
CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
    if (bytes) *bytes = 0;
    return CUDA_ERROR_INVALID_DEVICE;
}
CUresult cuCtxCreate(CUcontext *ctx, unsigned int flags, CUdevice dev) {
    return CUDA_ERROR_INVALID_DEVICE;
}
CUresult cuCtxDestroy(CUcontext ctx) { return CUDA_SUCCESS; }
CUresult cuCtxPushCurrent(CUcontext ctx) { return CUDA_SUCCESS; }
CUresult cuCtxPopCurrent(CUcontext *ctx) { if (ctx) *ctx = NULL; return CUDA_SUCCESS; }
CUresult cuCtxSynchronize(void) { return CUDA_SUCCESS; }
CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) { return CUDA_ERROR_NO_DEVICE; }
CUresult cuMemFree(CUdeviceptr dptr) { return CUDA_SUCCESS; }
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    return CUDA_ERROR_NO_DEVICE;
}
CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    return CUDA_ERROR_NO_DEVICE;
}
CUresult cuModuleLoad(CUmodule *module, const char *fname) { return CUDA_ERROR_NO_DEVICE; }
CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    return CUDA_ERROR_NO_DEVICE;
}
CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                        unsigned int gridDimZ, unsigned int blockDimX,
                        unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream,
                        void **kernelParams, void **extra) {
    return CUDA_ERROR_NO_DEVICE;
}
CUresult cuStreamCreate(CUstream *phStream, unsigned int flags) { return CUDA_ERROR_NO_DEVICE; }
CUresult cuStreamDestroy(CUstream hStream) { return CUDA_SUCCESS; }
CUresult cuStreamSynchronize(CUstream hStream) { return CUDA_SUCCESS; }
CUresult cuGetErrorString(CUresult error, const char **pStr) {
    if (pStr) *pStr = "CUDA stub: no device";
    return CUDA_SUCCESS;
}
CUresult cuGetErrorName(CUresult error, const char **pStr) {
    if (pStr) *pStr = "CUDA_STUB_ERROR";
    return CUDA_SUCCESS;
}

/* NVML stubs */
int nvmlInit(void) { return 0; }
int nvmlShutdown(void) { return 0; }
int nvmlDeviceGetCount(unsigned int *count) { if(count) *count=0; return 0; }