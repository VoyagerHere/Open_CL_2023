#include "../include/matrix_mult.h"

#include <omp.h>
#include <cstring> 
#include <iostream> 

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

cl_uint getPlatforms(std::vector<cl_platform_id>& pl) {
  cl_uint platformCount = 0;
  clGetPlatformIDs(0, nullptr, &platformCount);

  if (platformCount == 0) {
    throw -1;
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

void mult_seq(const size_t m, const size_t n, const size_t k,
                           const float* a, const float* b, float* c) {
  std::memset(c, 0, n * k * sizeof(float));

  for (auto i = 0; i < m; ++i) {
    for (int l = 0; l < n; ++l) {
      for (auto j = 0; j < k; ++j) {
        c[i * k + j] += a[i * n + l] * b[l * k + j];
      }
    }
  }
}

void mult_omp(const size_t m, const size_t n, const size_t k,
                               const float* a, const float* b, float* c) {
  std::memset(c, 0, n * k * sizeof(float));

#pragma omp parallel for shared(a, b, c) num_threads(THREADS)
  for (int i = 0; i < static_cast<int>(m); ++i) {
    for (int l = 0; l < static_cast<int>(n); ++l) {
      for (int j = 0; j < static_cast<int>(k); ++j) {
        c[i * k + j] += a[i * n + l] * b[l * k + j];
      }
    }
  }
}

double mult_gpu(
    const size_t m, const size_t n, const size_t k, const float* a,
    const float* b, float* c,
    std::pair<cl_platform_id, cl_device_id>& dev_pair) {
  cl_int error = CL_SUCCESS;
  cl_context_properties properties[3] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)dev_pair.first, 0};

  cl_context context = clCreateContext(properties, 1, &dev_pair.second, nullptr,
                                       nullptr, &error);

  cl_command_queue queue =
      clCreateCommandQueue(context, dev_pair.second, 0, NULL);

  cl_program program = createProgramFromSource(context, "kernels/normal.cl");
  clBuildProgram(program, 1, &dev_pair.second, nullptr, nullptr, nullptr);

  cl_kernel kernel = clCreateKernel(program, "normal", &error);

  cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                   sizeof(float) * m * n, nullptr, &error);
  cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                   sizeof(float) * n * k, nullptr, &error);
  cl_mem c_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                   sizeof(float) * m * k, nullptr, &error);

  clEnqueueWriteBuffer(queue, a_buffer, CL_TRUE, 0, sizeof(float) * m * n, a, 0,
                       nullptr, nullptr);
  clEnqueueWriteBuffer(queue, b_buffer, CL_TRUE, 0, sizeof(float) * n * k, b, 0,
                       nullptr, nullptr);

  clSetKernelArg(kernel, 0, sizeof(unsigned int), &m);
  clSetKernelArg(kernel, 1, sizeof(unsigned int), &n);
  clSetKernelArg(kernel, 2, sizeof(unsigned int), &k);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), &a_buffer);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), &b_buffer);
  clSetKernelArg(kernel, 5, sizeof(cl_mem), &c_buffer);

  size_t group = 16;
  clGetKernelWorkGroupInfo(kernel, dev_pair.second, CL_KERNEL_WORK_GROUP_SIZE,
                           sizeof(size_t), &group, nullptr);

  size_t* global = new size_t[2];
  global[0] = m;
  global[1] = k;

  size_t* local = new size_t[2];
  local[0] = static_cast<size_t>(BLOCK);
  local[1] = static_cast<size_t>(BLOCK);

  const size_t ndims = 2;
  auto t0 = omp_get_wtime();
  clEnqueueNDRangeKernel(queue, kernel, ndims, nullptr, global, local, 0,
                         nullptr, nullptr);
  clFinish(queue);
  auto t1 = omp_get_wtime();

  clEnqueueReadBuffer(queue, c_buffer, CL_TRUE, 0, sizeof(float) * m * k, c, 0,
                      nullptr, nullptr);

  clReleaseMemObject(a_buffer);
  clReleaseMemObject(b_buffer);
  clReleaseMemObject(c_buffer);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return t1 - t0;
}

double mult_gemm(const size_t m, const size_t n, const size_t k, const float* a,
            const float* b, float* c,
            std::pair<cl_platform_id, cl_device_id>& dev_pair) {
  cl_int error = CL_SUCCESS;
  cl_context_properties properties[3] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)dev_pair.first, 0};

  cl_context context = clCreateContext(properties, 1, &dev_pair.second, nullptr,
                                       nullptr, &error);

  cl_command_queue queue =
      clCreateCommandQueue(context, dev_pair.second, 0, NULL);

  cl_program program = createProgramFromSource(context, "kernels/gemm.cl");
  std::string build_options = "-DBLOCK=" + std::to_string(BLOCK);
  clBuildProgram(program, 1, &dev_pair.second, build_options.c_str(), nullptr,
                 nullptr);

  cl_kernel kernel = clCreateKernel(program, "gemm", &error);

  cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                   sizeof(float) * m * n, nullptr, &error);
  cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                   sizeof(float) * n * k, nullptr, &error);
  cl_mem c_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                   sizeof(float) * m * k, nullptr, &error);

  clEnqueueWriteBuffer(queue, a_buffer, CL_TRUE, 0, sizeof(float) * m * n, a, 0,
                       nullptr, nullptr);
  clEnqueueWriteBuffer(queue, b_buffer, CL_TRUE, 0, sizeof(float) * n * k, b, 0,
                       nullptr, nullptr);

  clSetKernelArg(kernel, 0, sizeof(unsigned int), &m);
  clSetKernelArg(kernel, 1, sizeof(unsigned int), &n);
  clSetKernelArg(kernel, 2, sizeof(unsigned int), &k);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), &a_buffer);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), &b_buffer);
  clSetKernelArg(kernel, 5, sizeof(cl_mem), &c_buffer);

  size_t group = 16;
  clGetKernelWorkGroupInfo(kernel, dev_pair.second, CL_KERNEL_WORK_GROUP_SIZE,
                           sizeof(size_t), &group, nullptr);

  size_t* global = new size_t[2];
  global[0] = k;
  global[1] = m;

  size_t* local = new size_t[2];
  local[0] = static_cast<size_t>(BLOCK);
  local[1] = static_cast<size_t>(BLOCK);

  const size_t ndims = 2;
  auto t0 = omp_get_wtime();
  clEnqueueNDRangeKernel(queue, kernel, ndims, nullptr, global, local, 0,
                         nullptr, nullptr);
  clFinish(queue);
  auto t1 = omp_get_wtime();

  clEnqueueReadBuffer(queue, c_buffer, CL_TRUE, 0, sizeof(float) * m * k, c, 0,
                      nullptr, nullptr);

  clReleaseMemObject(a_buffer);
  clReleaseMemObject(b_buffer);
  clReleaseMemObject(c_buffer);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return t1 - t0;
}

double mult_gemm_image(
  const size_t m, const size_t n, const size_t k, float* a,
  float* b, float* c,
  std::pair<cl_platform_id, cl_device_id>& dev_pair) {
 cl_int error = CL_SUCCESS;
 cl_context_properties properties[3] = {
    CL_CONTEXT_PLATFORM, (cl_context_properties)dev_pair.first, 0};

 cl_context context = clCreateContext(properties, 1, &dev_pair.second, nullptr,
                                 nullptr, &error);

 cl_command_queue queue =
    clCreateCommandQueue(context, dev_pair.second, 0, NULL);

 cl_program program = createProgramFromSource(context, "kernels/gemm_image.cl");
 std::string build_options = "-DBLOCK=" + std::to_string(BLOCK);
 clBuildProgram(program, 1, &dev_pair.second, build_options.c_str(), nullptr,
               nullptr);

 cl_kernel kernel = clCreateKernel(program, "gemm_image", &error);

 // Create image descriptors
 cl_image_desc desc;
 memset(&desc, 0, sizeof(desc));
 desc.image_type = CL_MEM_OBJECT_IMAGE2D;
 desc.image_width = m;
 desc.image_height = n;
 desc.image_row_pitch = sizeof(float) * m;
 desc.image_slice_pitch = sizeof(float) * m * n;

 cl_image_format format;
 format.image_channel_order = CL_R;
 format.image_channel_data_type = CL_FLOAT;

 // Create images
 cl_mem a_image = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, &desc, a, &error);
 cl_mem b_image = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, &desc, b, &error);
 cl_mem c_image = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &desc, nullptr, &error);

 // Set kernel arguments
 clSetKernelArg(kernel, 0, sizeof(unsigned int), &m);
 clSetKernelArg(kernel, 1, sizeof(unsigned int), &n);
 clSetKernelArg(kernel, 2, sizeof(unsigned int), &k);
 clSetKernelArg(kernel, 3, sizeof(cl_mem), &a_image);
 clSetKernelArg(kernel, 4, sizeof(cl_mem), &b_image);
 clSetKernelArg(kernel, 5, sizeof(cl_mem), &c_image);

 size_t group = 16;
 clGetKernelWorkGroupInfo(kernel, dev_pair.second, CL_KERNEL_WORK_GROUP_SIZE,
                       sizeof(size_t), &group, nullptr);

 size_t global[2] = {m, n};
 size_t local[2] = {static_cast<size_t>(BLOCK), static_cast<size_t>(BLOCK)};

 const size_t ndims = 2;
 auto t0 = omp_get_wtime();
 clEnqueueNDRangeKernel(queue, kernel, ndims, nullptr, global, local, 0,
                     nullptr, nullptr);
 clFinish(queue);
 auto t1 = omp_get_wtime();
 size_t origin[3] = {0, 0, 0};
 size_t region[3] = {m, n, 1};

 clEnqueueReadImage(queue, c_image, CL_TRUE, origin, region, 0, 0, c, 0, nullptr, nullptr);

 clReleaseMemObject(a_image);
 clReleaseMemObject(b_image);
 clReleaseMemObject(c_image);
 clReleaseProgram(program);
 clReleaseKernel(kernel);
 clReleaseCommandQueue(queue);
 clReleaseContext(context);

 return t1 - t0;
}

