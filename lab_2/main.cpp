#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

#include "include/axpy.h"
#include "include/utils.h"

#define SINGLE_EPS 1e-04
#define DOUBLE_EPS 1e-13

int main(int argc, char** argv) {
  std::vector<cl_platform_id> platforms;
  cl_uint platformCount = getCountAndListOfPlatforms(platforms);
  std::vector<std::pair<cl_platform_id, cl_device_id>> gpus, cpus;
  if (platformCount < 1) {
    return -1;
  }
  std::cout << platforms.size() << std::endl;

  for (size_t i = 0; i < platformCount; i++) {
    cl_platform_id platform = platforms[i];
    cl_device_id gpu = getDevice(CL_DEVICE_TYPE_GPU, platform);
    if (gpu != nullptr) gpus.push_back(std::make_pair(platform, gpu));

    cl_device_id cpu = getDevice(CL_DEVICE_TYPE_CPU, platform);
    if (cpu != nullptr) cpus.push_back(std::make_pair(platform, cpu));
  }

  std::cout << cpus.size();
  const int n = 150'000'000;
  const int inc_x = 1;
  const int inc_y = 1;

  const int x_size = n * inc_x;
  const int y_size = n * inc_y;

  std::cout << "saxpy" << std::endl << std::endl;
  {
    float a = 10.0;
    float* x = new float[x_size];
    float* y = new float[y_size];
    float* y_ref = new float[y_size];
    float* result_ref = new float[y_size];
    fillData<float>(x, x_size);

    // SEQ
    fillData<float>(y, y_size);
    std::memcpy(y_ref, y, y_size * sizeof(float));
    auto t0 = omp_get_wtime();
    saxpy(n, a, x, inc_x, y, inc_y);
    auto t1 = omp_get_wtime();
    std::cout << "SEQ: " << t1 - t0 << std::endl;
    std::memcpy(result_ref, y, y_size * sizeof(float));

    // OMP
    std::memcpy(y, y_ref, y_size * sizeof(float));
    t0 = omp_get_wtime();
    saxpy_omp(n, a, x, inc_x, y, inc_y);
    t1 = omp_get_wtime();
    std::cout << "OMP: " << t1 - t0 << " "
              << check<float>(result_ref, y, y_size, SINGLE_EPS) << std::endl;


    std::cout << "OpenCl CPU"
              << "\n";
    for (size_t group_size = 8; group_size <= 256; group_size *= 2) {
      // CPU OPENCL
      for (size_t i = 0; i < cpus.size(); i++) {
        std::memcpy(y, y_ref, y_size * sizeof(float));
        double time = saxpy_cl(n, a, x, inc_x, y, inc_y, gpus[i], group_size);
        char name[128];
        clGetDeviceInfo(cpus[i].second, CL_DEVICE_NAME, 128, name, nullptr);
        std::cout << "Group size: " << group_size
                  << "OpenCL CPU: " << time << " "
                  << check<float>(result_ref, y, y_size, SINGLE_EPS) << std::endl;
      }
    }

    std::cout << "GPU"
              << "\n";
    for (size_t group_size = 8; group_size <= 256; group_size *= 2) {
      // GPU OPENCL
      for (size_t i = 0; i < gpus.size(); i++) {
        std::memcpy(y, y_ref, y_size * sizeof(float));

        double time = saxpy_cl(n, a, x, inc_x, y, inc_y, gpus[i], group_size);
        char name[128];
        clGetDeviceInfo(gpus[i].second, CL_DEVICE_NAME, 128, name, nullptr);
        std::cout << "Group size: " << group_size
                  << "GPU: " << time << " "
                  << check<float>(result_ref, y, y_size, SINGLE_EPS) << std::endl;
      }
    }

    delete[] x;
    delete[] y;
    delete[] y_ref;
    delete[] result_ref;
  }

  std::cout << std::endl << "daxpy" << std::endl << std::endl;
  {
    double a = 10.0;
    double* x = new double[x_size];
    double* y = new double[y_size];
    double* y_ref = new double[y_size];
    double* result_ref = new double[y_size];
    fillData<double>(x, x_size);

    fillData<double>(y, y_size);
    std::memcpy(y_ref, y, y_size * sizeof(double));
    auto t0 = omp_get_wtime();
    daxpy(n, a, x, inc_x, y, inc_y);
    auto t1 = omp_get_wtime();
    std::cout << "SEQ: " << t1 - t0 << std::endl;
    std::memcpy(result_ref, y, y_size * sizeof(double));

    std::memcpy(y, y_ref, y_size * sizeof(double));
    t0 = omp_get_wtime();
    daxpy_omp(n, a, x, inc_x, y, inc_y);
    t1 = omp_get_wtime();
    std::cout << "OMP: " << t1 - t0 << " "
              << check<double>(result_ref, y, y_size, DOUBLE_EPS) << std::endl;
   
   
    std::cout << "OpenCL CPU"
              << "\n";
    for (size_t group_size = 8; group_size <= 256; group_size *= 2) {
      for (size_t i = 0; i < cpus.size(); i++) {
        std::memcpy(y, y_ref, y_size * sizeof(double));
        double time = daxpy_cl(n, a, x, inc_x, y, inc_y, gpus[i]);
        char name[128];
        clGetDeviceInfo(cpus[i].second, CL_DEVICE_NAME, 128, name, nullptr);
        std::cout << "Size: " << group_size
                  << " GPU: " << time << " "
                  << check<double>(result_ref, y, y_size, DOUBLE_EPS) << std::endl;
      }
    }
    std::cout << "GPU"
              << "\n";
    for (size_t group_size = 8; group_size <= 256; group_size *= 2) {
      for (size_t i = 0; i < gpus.size(); i++) {
        std::memcpy(y, y_ref, y_size * sizeof(double));
        double time = daxpy_cl(n, a, x, inc_x, y, inc_y, gpus[i]);
        char name[128];
        clGetDeviceInfo(gpus[i].second, CL_DEVICE_NAME, 128, name, nullptr);
        std::cout << "Size: " << group_size
                  << " GPU: " << time << " "
                  << check<double>(result_ref, y, y_size, DOUBLE_EPS) << std::endl;
      }
    }

    delete[] x;
    delete[] y;
    delete[] y_ref;
    delete[] result_ref;
  }

  return 0;
}
