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

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

template<typename TilingConfig, typename SparseKernelConfig>
static void SpMM_SplitK_Kernel_Ex(cudaStream_t stream,
                                  const half*  A,
                                  const uint4* Compressed_A,
                                  const int*   TileOffsets,
                                  const half*  B,
                                  half*        Reduction_Workspace,
                                  const int    M_Global,
                                  const int    N_Global,
                                  const int    K_Global,
                                  int          Split_K);

/*
half* Reduction_Workspace:  1. Requiring an extra memory space in device memory for un-reducted intermediate output
tensors
                            2. Reduction_Workspace_Size = max( Split_K * M_Global * N_Global ) * sizeof(fp16)
int Split_K:                Split K dimension into Split_K Parts
*/
cudaError_t SpMM_SplitK_API(cudaStream_t stream,
                            const half*  A,
                            const uint4* Compressed_A,
                            const int*   TileOffsets,
                            const half*  B,
                            half*        C,
                            const int    M_Global,
                            const int    N_Global,
                            const int    K_Global,
                            half*        Reduction_Workspace,  // Identical workspace for all SpMM kernel launches
                            int          Split_K);

// Generating Tiled-CSL format from dense format
__host__ int InitSparseMatrixA_API(half* A_h, int M, int N, int K, uint32_t** Compressed_A, int** TileOffsets);
// Generating Tiled-CSL format from dense format, without the optimization named "Ahead of Time Sparse Data Reordering"
__host__ int InitSparseMatrixA_API_NoReorder(half*      A_h,
                                             int        M,
                                             int        N,
                                             int        K,
                                             uint32_t** Compressed_A,  // CPU PTR
                                             int**      TileOffsets);       // CPU_PTR

// Used by ft-tools
extern "C" void GenSparseMatrixBinFile(char* DenseMatrixFileName,
                                       int   M,
                                       int   N,
                                       int   K,
                                       char* NZWeightsFileName,
                                       char* TileOffsetsFileName,
                                       char* OutputSizesFileName);