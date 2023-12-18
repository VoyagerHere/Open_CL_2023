#ifndef _UTILS_H
#define _UTILS_H

#include <CL/cl.h>


#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <ctime>

cl_program createProgramFromSource(cl_context ctx, const char* file);
cl_uint getCountAndListOfPlatforms(std::vector<cl_platform_id>& pl);
cl_device_id getDevice(cl_device_type type, cl_platform_id& plfrm_id);

#define TIME_US(t0, t1) std::chrono::duration_cast<us>(t1 - t0).count() << " us"
#define TIME_MS(t0, t1) std::chrono::duration_cast<ms>(t1 - t0).count() << " ms"
#define TIME_S(t0, t1) std::chrono::duration_cast<s>(t1 - t0).count() << " s"

using us = std::chrono::microseconds;
using ms = std::chrono::milliseconds;
using s = std::chrono::seconds;

using timer = std::pair<std::chrono::high_resolution_clock::time_point,
                        std::chrono::high_resolution_clock::time_point>;

template <typename T>
void fillData(T* data, const size_t size) {
  srand(time(0));
  for (int i = 0; i < size; i++) 
    data[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
}

template <typename T>
bool checkCorrect(T* current, T* reference, int size) {
  std::cout << std::fixed;
  std::cout.precision(6);
  bool correct = false;
  for (int i = 0; i < size; ++i) {
    if (std::abs(current[i] - reference[i]) >= std::numeric_limits<T>::epsilon()) {
      std::cout << "In index" << i << " expected: " << reference[i] << " current: " << current[i] << std::endl;
      correct = false;
      break;
    }
    correct = true;
  }
  return correct;
}

template <typename T>
std::string check(T* result, T* reference, size_t size) {
  bool res = checkCorrect<T>(result, reference, size);
  if (res) {
    return "PASSED";
  }
  return "FAILED";
}

cl_uint getCountAndListOfPlatforms(std::vector<cl_platform_id>& pl) {
  cl_uint platformCount = 0;
  clGetPlatformIDs(0, nullptr, &platformCount);

  if (platformCount == 0) {
    return -1;
  }

  cl_platform_id* platforms = new cl_platform_id[platformCount];
  clGetPlatformIDs(platformCount, platforms, nullptr);
  for (cl_uint i = 0; i < platformCount; ++i) {
    char platformName[128];
    clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, platformName,
                      nullptr);
    cl_uint cpuCount = 0;
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 0, nullptr, &cpuCount);
    cl_device_id* cpus = new cl_device_id[cpuCount];
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, cpuCount, cpus, nullptr);
    for (cl_uint j = 0; j < cpuCount; ++j) {
      char cpuName[128];
      clGetDeviceInfo(cpus[j], CL_DEVICE_NAME, 128, cpuName, nullptr);
    }

    cl_uint gpuCount = 0;
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &gpuCount);
    cl_device_id* gpus = new cl_device_id[gpuCount];
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, gpuCount, gpus, nullptr);
    for (cl_uint j = 0; j < gpuCount; ++j) {
      char gpuName[128];
      clGetDeviceInfo(gpus[j], CL_DEVICE_NAME, 128, gpuName, nullptr);
    }
    pl.push_back(platforms[i]);
  }

  delete[] platforms;
  return platformCount;
}

cl_program createProgramFromSource(cl_context ctx, const char* file) {
  std::fstream kernel_file(file, std::ios::in);
  std::string kernel_code((std::istreambuf_iterator<char>(kernel_file)),
                          std::istreambuf_iterator<char>());
  kernel_file.close();
  const char* kernel_code_p = kernel_code.c_str();
  size_t kernel_code_len = kernel_code.size();

  cl_int errorcode = CL_SUCCESS;
  cl_program program = clCreateProgramWithSource(ctx, 1, &kernel_code_p,
                                                 &kernel_code_len, &errorcode);

  return program;
}

cl_device_id getDevice(cl_device_type type, cl_platform_id& plfrm_id) {
  cl_uint device_count = 0;
  clGetDeviceIDs(plfrm_id, type, 0, nullptr, &device_count);

  if (device_count == 0) return nullptr;

  if (device_count > 0) {
    std::vector<cl_device_id> device_vec(device_count);
    clGetDeviceIDs(plfrm_id, type, device_count, device_vec.data(), nullptr);

    if (device_vec.size() > 0) {
      cl_device_id id = device_vec.front();
      return id;
    }
  }
  return nullptr;
}
#endif // UTILS_H