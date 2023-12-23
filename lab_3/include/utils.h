#ifndef _UTILS_H
#define _UTILS_H

#include <CL/cl.h>


#include <cstring>
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>


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
  for (size_t i = 0; i < size; i++) 
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

#endif // UTILS_H