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

#include "./sputnik_utils.h"
#include "sputnik/sputnik.h"
#include <cusparseLt.h>  // cusparseLt header

#define SPARTA_M 2
#define SPARTA_N 4

void transform(half* A_h, half* A1_h, half* A2_h, int length)
{
    // split the matrix A into A1 and A2
    // A1 is for the saprse tensor core, A2 is for the finegrained sparse kernel
    memset(A1_h, 0, sizeof(half) * length);
    memset(A2_h, 0, sizeof(half) * length);
    assert(length % SPARTA_N == 0);
    int nnz = 0;
    for (int i = 0; i < length / SPARTA_N; i++) {
        int start = i * SPARTA_N;
        int end   = start + SPARTA_N;
        nnz       = 0;
        for (int j = start; j < end; j++) {
            if (fabs(__half2float(A_h[j])) > 0.0000001) {
                if (nnz < SPARTA_M) {
                    A1_h[j] = A_h[j];
                }
                else {
                    A2_h[j] = A_h[j];
                }
                nnz++;
            }
        }
    }
    int NNZ_A  = 0;
    int NNZ_A1 = 0;
    int NNZ_A2 = 0;
    for (int i = 0; i < length; i++)
        if (fabs(__half2float(A_h[i])) > 0.0000001)
            NNZ_A++;
    for (int i = 0; i < length; i++)
        if (fabs(__half2float(A1_h[i])) > 0.0000001)
            NNZ_A1++;
    for (int i = 0; i < length; i++)
        if (fabs(__half2float(A2_h[i])) > 0.0000001)
            NNZ_A2++;
    printf("NZ_A: %2.3f \t NZ_A1: %2.3f \t NZ_A2: %2.3f\n",
           float(NNZ_A) / length,
           float(NNZ_A1) / length,
           float(NNZ_A2) / length);
}

void check_A1_h(half* A, int length)
{
    assert(length % 4 == 0);
    for (int i = 0; i < length / 4; i++) {
        int nnz = 0;
        for (int j = 0; j < 4; j++) {
            if (__half2float(A[i * 4 + j]) != 0.0f)
                nnz++;
        }
        if (nnz > 2) {
            printf("Failed to meet 2:4 requirements!\n");
            exit(-1);
        }
    }
    printf("Succeeded in meeting 2:4 requirements!\n");
}

// A, B, and C are row-major; A is the sparse matrix
int sparTA(half* A_h, half* B_h, half* C_h, int m, int n, int k, float* milliseconds)
{
    //
    half *A1_h, *A2_h;
    A1_h = (half*)malloc(sizeof(half) * m * k);
    if (A1_h == NULL) {
        printf("Error in sparTA.h: line %d malloc falied\n", __LINE__);
        exit(-1);
    }
    A2_h = (half*)malloc(sizeof(half) * m * k);
    if (A2_h == NULL) {
        printf("Error in sparTA.h: line %d malloc falied\n", __LINE__);
        exit(-1);
    }
    transform(A_h, A1_h, A2_h, m * k);
    // check_A1_h(A1_h, m*k);
    half *A1_d, *B_d, *C_d;
    CHECK_CUDA(cudaMalloc((void**)&A1_d, m * k * sizeof(half)))
    CHECK_CUDA(cudaMalloc((void**)&B_d, k * n * sizeof(half)))
    CHECK_CUDA(cudaMalloc((void**)&C_d, m * n * sizeof(half)))
    CHECK_CUDA(cudaMemcpy(A1_d, A1_h, m * k * sizeof(half), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(B_d, B_h, k * n * sizeof(half), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemset(C_d, 0, m * n * sizeof(half)))

    //--------------------------------------------------------------------------
    // cuSPARSELT
    cusparseLtHandle_t             handle;
    cusparseLtMatDescriptor_t      matA, matB, matC;
    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t         plan;
    CHECK_CUSPARSE(cusparseLtInit(&handle))
    // matrix descriptor initialization
    CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
        &handle, &matA, m, k, k, (unsigned int)16, CUDA_R_16F, CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT))
    CHECK_CUSPARSE(
        cusparseLtDenseDescriptorInit(&handle, &matB, k, n, n, (unsigned int)16, CUDA_R_16F, CUSPARSE_ORDER_ROW))
    CHECK_CUSPARSE(
        cusparseLtDenseDescriptorInit(&handle, &matC, m, n, n, (unsigned int)16, CUDA_R_16F, CUSPARSE_ORDER_ROW))
    // matmul, algorithm selection, and plan initialization
    CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(&handle,
                                                  &matmul,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &matA,
                                                  &matB,
                                                  &matC,
                                                  &matC,
                                                  CUSPARSE_COMPUTE_16F))
    CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT))
    CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel))
    //--------------------------------------------------------------------------
    // Prune the A matrix (in-place) and check the correcteness
    // CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, &matmul, dA, dA,
    //                                      CUSPARSELT_PRUNE_SPMMA_TILE, stream) )
    int* valid_d;
    CHECK_CUDA(cudaMalloc((void**)&valid_d, sizeof(int)))
    CHECK_CUSPARSE(cusparseLtSpMMAPruneCheck(&handle, &matmul, A1_d, valid_d, nullptr))
    int is_valid;
    CHECK_CUDA(cudaMemcpyAsync(&is_valid, valid_d, sizeof(int), cudaMemcpyDeviceToHost, nullptr))
    CHECK_CUDA(cudaStreamSynchronize(nullptr))
    if (is_valid != 0) {
        printf(
            "[Warning]: The matrix has been pruned in a wrong way. cusparseLtMatmul will not provide correct results\n");  // Maybe there is some problem within cusparseLtSpMMAPruneCheck();
        // return -1;
    }
    //--------------------------------------------------------------------------
    // Compress the A matrix
    half * A1_d_compressed, *A1_d_compressed_buffer;
    size_t compressed_size, compressBufferSize;
    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size, &compressBufferSize))
    CHECK_CUDA(cudaMalloc((void**)&A1_d_compressed, compressed_size))
    CHECK_CUDA(cudaMalloc((void**)&A1_d_compressed_buffer, compressBufferSize))
    CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle, &plan, A1_d, A1_d_compressed, A1_d_compressed_buffer, nullptr))
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Search the best kernel
    float alpha_sparTA = 1.0f;
    float beta_sparTA  = 1.0f;
    CHECK_CUSPARSE(cusparseLtMatmulSearch(
        &handle, &plan, &alpha_sparTA, A1_d_compressed, B_d, &beta_sparTA, C_d, C_d, nullptr, nullptr, 0))
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    size_t workspace_size;
    CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size))
    void* workspace_d;
    CHECK_CUDA(cudaMalloc((void**)&workspace_d, workspace_size))
    //--------------------------------------------------------------------------
    // Sputnik
    float* A_float_h = (float*)malloc(sizeof(float) * m * k);
    for (int i = 0; i < m * k; i++)
        A_float_h[i] = __half2float(A2_h[i]);
    sputnik_utils::SparseMatrix            sparse_matrix(m, k, A_float_h, sputnik_utils::IDENTITY, 4);
    sputnik_utils::CudaSparseMatrix<half2> sparse_matrix_gpu(sparse_matrix);
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Perform the matrix multiplication
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < WARM_UP_ITERATION; i++) {
        CUDA_CALL(sputnik::CudaSpmm(m,
                                    k,
                                    n,
                                    sparse_matrix_gpu.NumElementsWithPadding(),
                                    sparse_matrix_gpu.RowIndices(),
                                    sparse_matrix_gpu.Values(),
                                    sparse_matrix_gpu.RowOffsets(),
                                    sparse_matrix_gpu.ColumnIndices(),
                                    reinterpret_cast<half2*>(B_d),
                                    reinterpret_cast<half2*>(C_d),
                                    0));
        CHECK_CUSPARSE(cusparseLtMatmul(
            &handle, &plan, &alpha_sparTA, A1_d_compressed, B_d, &beta_sparTA, C_d, C_d, workspace_d, nullptr, 0))
    }
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATION; i++) {
        CUDA_CALL(sputnik::CudaSpmm(m,
                                    k,
                                    n,
                                    sparse_matrix_gpu.NumElementsWithPadding(),
                                    sparse_matrix_gpu.RowIndices(),
                                    sparse_matrix_gpu.Values(),
                                    sparse_matrix_gpu.RowOffsets(),
                                    sparse_matrix_gpu.ColumnIndices(),
                                    reinterpret_cast<half2*>(B_d),
                                    reinterpret_cast<half2*>(C_d),
                                    0));
        CHECK_CUSPARSE(cusparseLtMatmul(
            &handle, &plan, &alpha_sparTA, A1_d_compressed, B_d, &beta_sparTA, C_d, C_d, workspace_d, nullptr, 0))
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(milliseconds, start, stop);
    *milliseconds = *milliseconds / BENCHMARK_ITERATION;
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // destroy plan and handle
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matA))
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matB))
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matC))
    CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan))
    CHECK_CUSPARSE(cusparseLtDestroy(&handle))
    //--------------------------------------------------------------------------
    cudaFree(A1_d);
    cudaFree(B_d);
    cudaMemcpy(C_h, C_d, sizeof(half) * m * n, cudaMemcpyDeviceToHost);  // Col Major
    cudaFree(C_d);
    return 0;
}