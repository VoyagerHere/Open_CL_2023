#include <CL/cl.h>
#include <iostream>

int main() {
    const char * source = 
    " __kernel void square (                                      \n"\
    "                        __global float * input,              \n"\
    "                        __global float * output              \n"\
    "                        const unsigned int count             \n"\
    "                        ){                                   \n"\
    "          int i = get_global_id ( 0 );                       \n"\
    "                                                             \n"\
    "          if ( i < count )                                   \n"\
    "              output [i] = input [i] * input[i];             \n"\
    ")";

    cl_uint platformCount = 0;
    clGetPlatformIDs(0, nullptr, &platformCount);

    cl_platform_id* platforms = new cl_platform_id[platformCount];
    clGetPlatformIDs(platformCount, platforms, nullptr);

    cl_platform_id platform = platforms[0];

    cl_context_properties properties [3] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0
    };

    cl_context context = clCreateContextFromType (
      ( NULL == platform) ? NULL : properties,
      CL_DEVICE_TYPE_GPU,
      NULL,
      NULL,
      NULL
    );

    size_t size = 0;

    clGetContextInfo (
      context,
      CL_CONTEXT_DEVICES,
      0,
      NULL,
      &size
    );

    cl_command_queue queue = clCreateCommandQueue(
      context,
      device,
      0,
      NULL
    );

    size_t strlen [] = {strlen(source)};


    return 0;
}
