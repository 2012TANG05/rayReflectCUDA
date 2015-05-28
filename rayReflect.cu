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

    if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "?"))
    {
        printf("Usage : \n\n  device = n \n   n = 0 : run on GPU\n   n = 1 : run on CPU\n   default is 0(GPU)\n\n  theta = x : 天顶角方向分出x条射线,default x = 256\n  phi = y : 方位角方向分出y条射线,default y = 256 \n\n\n ");

        exit(EXIT_SUCCESS);
    }

    // By default, we use device 0, which GPU we use
    int devID = 0;

    int device = 0;
    int theta = 256;
    int phi = 256;
    int version = 0;

    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        device = getCmdLineArgumentInt(argc, (const char **)argv, "device");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "theta"))
    {
        theta = getCmdLineArgumentInt(argc, (const char **)argv, "theta");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "phi"))
    {
        phi = getCmdLineArgumentInt(argc, (const char **)argv, "phi");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "version"))
    {
        version = getCmdLineArgumentInt(argc, (const char **)argv, "version");
    }

    cudaError_t error;
    error = cudaGetDevice(&devID);                                                                                                                                 
                                                                                                                                                                   
    if (error != cudaSuccess)                                                                                                                                      
    {                                                                                                                                                              
        printf("Usage : \n\n  device = n \n   n = 0 : run on GPU\n   n = 1 : run on CPU\n   default is 0(GPU)\n\n  theta = x : 天顶角方向分出x条射线,default x = 256\n  phi = y : 方位角方向分出y条射线,default y = 256\n\n\n ");
        printf("cudaGetDevice returned error code %d, line(%d)\n\n", error, __LINE__);                                                                               
    }                                                                                                                                                              

    int result; 

   printf("n= %d", device);

    //调用GPU的函数
    if (device == 0)
    {
        result = rayReflectGPU(argc, argv, theta, phi, version);
    }
    //调用CPU函数
    else if (device == 1)
    {
        result = rayReflectCPU(argc,argv,theta,phi);
    }
    else
    {
        result = 1;
    }                                                                                                                                                   
    exit(result);
}
