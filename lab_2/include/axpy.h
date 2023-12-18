#ifndef _AXPY_H_
#define _AXPY_H_

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.h>

#include "utils.h"

#include <omp.h>
#include <vector>

cl_uint getCountAndListOfPlatforms(std::vector<cl_platform_id>& pl);
cl_program createProgramFromSource(cl_context ctx, const char* file);
cl_device_id getDevice(cl_device_type type, cl_platform_id& plfrm_id);

void saxpy(const int& n, const float a, const float* x, const int& incx,
           float* y, const int& incy);
void daxpy(const int& n, const double a, const double* x, const int& incx,
           double* y, const int& incy);

void saxpy_omp(const int& n, const float a, const float* x, const int& incx,
               float* y, const int& incy);
void daxpy_omp(const int& n, const double a, const double* x, const int& incx,
               double* y, const int& incy);

void saxpy_cl(int n, float a, const float* x, int incx, float* y, int incy,
              std::pair<cl_platform_id, cl_device_id>& dev_pair, timer& time,
              size_t group_size = 256);
void daxpy_cl(int n, double a, const double* x, int incx, double* y, int incy,
              std::pair<cl_platform_id, cl_device_id>& dev_pair, timer& time,
              size_t group_size = 256);
#endif // _AXPY_H_