#ifndef _MATRIX_MULT_H
#define _MATRIX_MULT_H

#define THREADS 8
#define BLOCK 16

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "CL/cl.h"

void mult_seq(const size_t m, const size_t n, const size_t k,
                           const float* a, const float* b, float* c);
void mult_omp(const size_t m, const size_t n, const size_t k,
                               const float* a, const float* b, float* c);
double mult_gpu(
    const size_t m, const size_t n, const size_t k, const float* a,
    const float* b, float* c,
    std::pair<cl_platform_id, cl_device_id>& dev_pair);

double mult_gemm(const size_t m, const size_t n, const size_t k, const float* a,
            const float* b, float* c,
            std::pair<cl_platform_id, cl_device_id>& dev_pair);

cl_program createProgramFromSource(cl_context ctx, const char* file);
cl_uint getPlatforms(std::vector<cl_platform_id>& pl);
cl_device_id getDevice(cl_device_type type, cl_platform_id& plfrm_id);

template <typename T>
void fillMatrix(T* data, const size_t size) {
  for (int i = 0; i < size; ++i) data[i] = i;
}
#endif // _MATRIX_MULT_H
