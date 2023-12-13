#ifndef _AXPY_H_
#define _AXPY_H_

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.h>
#include <omp.h>
#include <vector>


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