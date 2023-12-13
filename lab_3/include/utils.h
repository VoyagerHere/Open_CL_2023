#ifndef _UTILS_H
#define _UTILS_H

#include <CL/cl.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <ctime>

cl_program createProgramFromSource(cl_context ctx, const char* file);
cl_uint getCountAndListOfPlatforms(std::vector<cl_platform_id>& pl);
cl_device_id getDevice(cl_device_type type, cl_platform_id& plfrm_id);


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
  for (int i = 0; i < size; ++i) {
    if (std::abs(current[i] - reference[i]) >= std::numeric_limits<T>::epsilon()) {
      std::cout << "In index" << i << " expected: " << reference[i] << " current: " << current[i] << std::endl;
      return false;
    }
  return true;
  }
}

template <typename T>
std::string check(bool flag, T* result, T* reference, size_t size) {
  if (flag) {
    bool res = checkCorrect<T>(result, reference, size);
    if (res) {
      return "PASSED";
    }
    return "FAILED";
  }
}


#define TIME_US(t0, t1) std::chrono::duration_cast<us>(t1 - t0).count() << " us"
#define TIME_MS(t0, t1) std::chrono::duration_cast<ms>(t1 - t0).count() << " ms"
#define TIME_S(t0, t1) std::chrono::duration_cast<s>(t1 - t0).count() << " s"

using us = std::chrono::microseconds;
using ms = std::chrono::milliseconds;
using s = std::chrono::seconds;

using timer = std::pair<std::chrono::high_resolution_clock::time_point,
                        std::chrono::high_resolution_clock::time_point>;
#endif // UTILS_H