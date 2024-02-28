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

template <typename T>
void fillData(T* data, const size_t size) {
  srand(time(0));
  for (size_t i = 0; i < size; i++) 
    data[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
}

template <typename T>
bool checkCorrect(T* current, T* reference, int size, T EPS) {
  // std::cout << std::fixed;
  // std::cout.precision(6);
  bool correct = false;
  for (int i = 0; i < size; ++i) {
    if (std::abs(current[i] - reference[i]) >= EPS) {
      std::cout << "In index" << i << " expected: " << reference[i] << " current: " << current[i] << std::endl;
      correct = false;
      break;
    }
    correct = true;
  }
  return correct;
}

template <typename T>
std::string check(T* result, T* reference, size_t size, T EPS) {
  bool res = checkCorrect<T>(result, reference, size, EPS);
  if (res) {
    return "PASSED";
  }
  return "FAILED";
}

#endif // UTILS_H