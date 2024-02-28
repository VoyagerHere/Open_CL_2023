#include <CL/cl.h>
#include <stdio.h>
#include <iostream>

const char* kernel_1 =
"__kernel void information() {                                             "
"                       \n"
"    int groupId = get_group_id(0);                                        "
"                       \n"
"    int localId = get_local_id(0);                                        "
"                       \n"
"    int globalId = get_global_id(0);                                      "
"                       \n"
"    printf(\"I am from %d block, %d thread (global index: %d)\\n\", groupId, "
"localId, globalId);    \n"
"}                                                                         "
"                       \n";

const char* kernel_2 =
"__kernel void calculate(__global int* a) {    \n"
"   int globalId = get_global_id(0);           \n"
"   a[globalId] = a[globalId] + globalId;      \n"
"}                                             \n";

int main() {
  cl_uint numPlatforms = 0;
  clGetPlatformIDs(0, NULL, &numPlatforms);
  cl_platform_id platform = NULL;

  if (numPlatforms > 0) {
    cl_platform_id* platforms = new cl_platform_id[numPlatforms];

    clGetPlatformIDs(numPlatforms, platforms, NULL);

    // printf("Available platforms:\n");
    for (cl_uint i = 0; i < numPlatforms; i++) {
      char platformName[128];
      clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, platformName,
        NULL);
      printf("%d - %s\n", i, platformName);
    }
    printf("\n");

    platform = platforms[0];
    delete[] platforms;
  }

  cl_uint numDevices = 0;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
  cl_device_id device = NULL;

  if (numDevices > 0) {
    cl_device_id* devices = new cl_device_id[numDevices];

    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);

    // printf("Available devices:\n");
    for (cl_uint i = 0; i < numDevices; i++) {
      char deviceName[128];
      clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 128, deviceName, NULL);
      // printf("%d - %s\n", i, deviceName);
    }
    printf("\n");

    device = devices[0];
    delete[] devices;
  }

  cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM,
                                         (cl_context_properties)platform, 0 };
  cl_context context =
    clCreateContext(properties, 1, &device, NULL, NULL, NULL);

  cl_command_queue queue =
      clCreateCommandQueue(context, device, 0, NULL);

  {
    cl_program program =
      clCreateProgramWithSource(context, 1, &kernel_1, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "information", NULL);

    size_t count = 20;
    size_t group = 5;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &count, &group, 0, NULL,
      NULL);

    clFlush(queue);
    clFinish(queue);

    clReleaseProgram(program);
    clReleaseKernel(kernel);
  }

  {
    size_t a_size = 8;
    cl_uint* a = (cl_uint*)malloc(a_size * sizeof(cl_uint));
    for (size_t i = 0; i < a_size; i++) {
      a[i] = i;
    }
    for (size_t i = 0; i < a_size; i++) {
      printf("%d ", a[i]);
    }
    printf("\n");
    cl_mem memory = clCreateBuffer(context, CL_MEM_READ_WRITE,
      a_size * sizeof(cl_uint), NULL, NULL);

    cl_program program =
      clCreateProgramWithSource(context, 1, &kernel_2, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "calculate", NULL);

    clEnqueueWriteBuffer(queue, memory, CL_TRUE, 0, a_size * sizeof(cl_uint), a,
      0, NULL, NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&memory);

    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &a_size, NULL, 0, NULL,
      NULL);
    clEnqueueReadBuffer(queue, memory, CL_TRUE, 0, a_size * sizeof(cl_uint), a,
      0, NULL, NULL);
    clFlush(queue);
    clFinish(queue);

    for (size_t i = 0; i < a_size; i++) {
      printf("%d ", a[i]);
    }

    clReleaseMemObject(memory);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
  }

  clReleaseContext(context);
  clReleaseCommandQueue(queue);

  return 0;
}
