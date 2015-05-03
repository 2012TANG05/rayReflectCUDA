#include <stdio.h>
#include <vector>
#include <ctime>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <string>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include "./lib/head.h"
using namespace std;

#define PI 3.1415926


int main(int argc, char **argv)
{
    printf("\n\n[Ray Reflect Using CUDA] - Starting...\n\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "?"))
    {
        printf("Usage : \n\n   device = n \n   n = 0 : run on GPU\n   n = 1 : run on CPU\n   default is 0(GPU)\n\n\n");

        exit(EXIT_SUCCESS);
    }

    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    int devID = 0;
    int device = 0;

    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        device = getCmdLineArgumentInt(argc, (const char **)argv, "device");
    }

    cudaError_t error;
    cudaDeviceProp deviceProp;                                                                                                                                     
    error = cudaGetDevice(&devID);                                                                                                                                 
                                                                                                                                                                   
    if (error != cudaSuccess)                                                                                                                                      
    {                                                                                                                                                              
        printf("cudaGetDevice returned error code %d, line(%d)\n\n", error, __LINE__);                                                                               
    }                                                                                                                                                              
                                                                                                                                                                   
    error = cudaGetDeviceProperties(&deviceProp, devID);                                                                                                           
                                                                                                                                                                   
    if (deviceProp.computeMode == cudaComputeModeProhibited)                                                                                                       
    {                                                                                                                                                              
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");                                         
        exit(EXIT_SUCCESS);                                                                                                                                        
    }                                                                                                                                                              
                                                                                                                                                                   
    if (error != cudaSuccess)                                                                                                                                      
    {                                                                                                                                                              
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);                                                                     
    }      
    else                                                                                                                                                           
    {                                                                                                                                                              
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);                             
    }                                                                                                                                                              
    
    int result; 

    //调用GPU的函数
    if (device == 0)
    {
        result = rayReflectGPU(argc,argv);
    }
    //调用CPU函数
    else if (device == 1)
    {
        result = rayReflectCPU(argc,argv);
    }
    else
    {
        printf("Usage : \n\n   device = n \n   n = 0 : run on GPU\n   n = 1 : run on CPU\n   default is 0(GPU)\n\n\n");
        result = 1;
    }                                                                                                                                                   
    exit(result);
}
