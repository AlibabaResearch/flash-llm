/***************************************************************************
 * Copyright 2023 The FLash-LLM Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

// Performance Benchmark
#define WARM_UP_ITERATION 100
#define BENCHMARK_ITERATION 10000
#ifdef USE_CUSPARSE
#define CUSPARSE_ITERATION 10
#endif
// Problem size
//#define M_GLOBAL    (36*1024)                               //
//#define K_GLOBAL    (9*1024)                                // must be 64X
//#define N_GLOBAL    16                                     // must be 8X
//#define SPLIT_K     3
//#define MATRIX_A_PRUNING_PERCENTAGE 80

//printf("\n\
//        CUSPARSE_SPMM_ALG_DEFAULT      = 0,\n\
//        CUSPARSE_SPMM_COO_ALG1         = 1,\n\
//        CUSPARSE_SPMM_COO_ALG2         = 2,\n\
//        CUSPARSE_SPMM_COO_ALG3         = 3,\n\
//        CUSPARSE_SPMM_COO_ALG4         = 5,\n\
//        CUSPARSE_SPMM_CSR_ALG1         = 4,\n\
//        CUSPARSE_SPMM_CSR_ALG2         = 6,\n\
//        CUSPARSE_SPMM_CSR_ALG3         = 12,\n\
//        CUSPARSE_SPMM_BLOCKED_ELL_ALG1 = 13\n");

void checkCublasError(cublasStatus_t status, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Cublas Error at line %d, Error Code: %d\n", line, status);
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUSPARSE(func)                                                                                           \
    {                                                                                                                  \
        cusparseStatus_t status = (func);                                                                              \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                                                       \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n",                                             \
                   __LINE__,                                                                                           \
                   cusparseGetErrorString(status),                                                                     \
                   status);                                                                                            \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    }
#define CHECK_CUDA(func)                                                                                               \
    {                                                                                                                  \
        cudaError_t status = (func);                                                                                   \
        if (status != cudaSuccess) {                                                                                   \
            printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, cudaGetErrorString(status), status);  \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    }

// void checkCusparseError(cusparseStatus_t status, int line) {
//    if (status != CUSPARSE_STATUS_SUCCESS) {
//        printf("CuSparse Error at line %d, Error Code: %d\n", line, status);
//        exit(EXIT_FAILURE);
//    }
//}

#define CUDA_CALL(code)                                                                                                \
    do {                                                                                                               \
        cudaError_t status = code;                                                                                     \
        std::string err    = cudaGetErrorString(status);                                                               \
        CHECK_EQ(status, cudaSuccess) << "CUDA Error: " << err;                                                        \
    } while (0)

void checkCudaError(cudaError_t error, int line)
{
    if (error != cudaSuccess) {
        printf("Cuda Error at line %d, Error Code: %d\n", line, error);
        exit(EXIT_FAILURE);
    }
}

void checkLastCudaError(int line)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Last Cuda Error Detected at line: %d, Error: %s.\n", line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

__host__ void
init_host_matrices(half* a, half* b, int M_GLOBAL, int K_GLOBAL, int N_GLOBAL, int MATRIX_A_PRUNING_PERCENTAGE)
{
    for (int i = 0; i < M_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            int r = rand() % 100;
            if (r >= MATRIX_A_PRUNING_PERCENTAGE)
                a[j + i * K_GLOBAL] = __float2half_rn(1.0f);
            else
                a[j + i * K_GLOBAL] = __float2half_rn(0.0f);
            // a[i] = __float2half_rn(0.0f);
        }
    }
    for (int i = 0; i < N_GLOBAL * K_GLOBAL; i++)
        b[i] = __float2half_rn(static_cast<float>((rand() % 5)) / 5 - 0.5f);
}

double ComputeTotalError(half* CuBlas, half* Other, int m, int n)
{
    double totalError = 0.0;
    for (int i = 0; i < m * n; i++)
        totalError += fabs(__half2float(CuBlas[i]) - __half2float(Other[i]));
    return totalError;
}

void PrintMismatch(const char* KernelName,
                   int         MaxNumMismatch,
                   float       ErrorThreshold,
                   half*       CuBlas,
                   half*       Other,
                   int         M_GLOBAL,
                   int         N_GLOBAL)
{
    printf("First 10 Mismatches between Cublas and %s:\n", KernelName);
    int count = 0;
    for (int i = 0; i < M_GLOBAL; i++) {
        for (int j = 0; j < N_GLOBAL; j++) {
            if (fabs(__half2float(CuBlas[i + j * M_GLOBAL]) - __half2float(Other[i + j * M_GLOBAL])) > ErrorThreshold) {
                count++;
                printf("(%d,%d) CuBlas=%f %s=%f\n",
                       i,
                       j,
                       __half2float(CuBlas[i + j * M_GLOBAL]),
                       KernelName,
                       __half2float(Other[i + j * M_GLOBAL]));
            }
            if (count == MaxNumMismatch)
                break;
        }
        if (count == MaxNumMismatch)
            break;
    }
}

void PrintPerformance(const char* KernelName, float milliseconds, float tflops, double error)
{
    printf("%-10s \t -> \t\t Time/ms: %5.3f \t Performance/TFLOPs: %4.2f \t TotalError: %.2lf\n",
           KernelName,
           milliseconds,
           tflops,
           error);
}
