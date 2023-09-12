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

#include "./MatMulUtilities.cuh"
#include "./Reduction_Kernel.cuh"
#include "./SpMM_Kernel.cuh"
#include <fstream>
#include <iostream>
#include <string>
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
                                  int          Split_K)
{
    static int SHMEM_SZ = max((TilingConfig::TILE_M * TILE_K + TilingConfig::TILE_N * TILE_K) * sizeof(half) * 2,
                              (TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C) * TilingConfig::TILE_N * sizeof(float));
    cudaFuncSetAttribute(
        SpMM_Kernel<TilingConfig, SparseKernelConfig>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    // printf("Max shared memory size: %d B\n", SHMEM_SZ);
    int dimN =
        max(N_Global / TilingConfig::TILE_N, 1);  // max(N_Global/TilingConfig::TILE_N,1) used when N=8, TILE_N=16
    int  dimM = M_Global * Split_K / TilingConfig::TILE_M;
    dim3 GridDim(dimN, dimM, 1);  // Grid Size is increased due to SplitK for higher SM occupancy
    dim3 BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1);
    //
    SpMM_Kernel<TilingConfig, SparseKernelConfig><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
        A, Compressed_A, TileOffsets, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);
}

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
                            int          Split_K)
{
#ifdef DEBUG_MODE
    printf(
        "SpMM_API.cu->SpMM_SplitK_API():  M: %d, N: %d, K: %d, SplitK: %d \n", M_Global, N_Global, K_Global, Split_K);
    assert(K_Global % TILE_K == 0);
    assert(M_Global % 256 == 0);
#endif
    half* SpMM_SplitK_OutputPTR;
    if (Split_K == 1)
        SpMM_SplitK_OutputPTR = C;
    else
        SpMM_SplitK_OutputPTR = Reduction_Workspace;
    // Batched SpMM
    switch (N_Global) {
        case 8:
            SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 1, 1>, SparseKernelConfig<64>>(
                stream, A, Compressed_A, TileOffsets, B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        case 16:
            SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 1>, SparseKernelConfig<64>>(
                stream, A, Compressed_A, TileOffsets, B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        case 32:
            SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 2>, SparseKernelConfig<64>>(
                stream, A, Compressed_A, TileOffsets, B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        case 64:
            // return SpMM_SplitK_Kernel_Ex< TilingConfig<4, 1, 4>, SparseKernelConfig<64> >
            SpMM_SplitK_Kernel_Ex<TilingConfig<2, 2, 2>, SparseKernelConfig<32>>(
                stream, A, Compressed_A, TileOffsets, B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        case 128:
            SpMM_SplitK_Kernel_Ex<TilingConfig<2, 2, 4>, SparseKernelConfig<32>>(
                stream, A, Compressed_A, TileOffsets, B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        default:
            if (N_Global % 128 == 0)
                SpMM_SplitK_Kernel_Ex<TilingConfig<2, 2, 4>, SparseKernelConfig<32>>(stream,
                                                                                     A,
                                                                                     Compressed_A,
                                                                                     TileOffsets,
                                                                                     B,
                                                                                     SpMM_SplitK_OutputPTR,
                                                                                     M_Global,
                                                                                     N_Global,
                                                                                     K_Global,
                                                                                     Split_K);
            else {
                printf("MM_Sparse_API Error: Unsupported N dimension %d!\n", N_Global);
                return cudaErrorUnknown;
            }
            break;
    }
    //
    cudaError_t Error = cudaGetLastError();
    if (Error != cudaSuccess)
        return Error;

    if (Split_K == 1)
        return Error;
    dim3 GridDim((M_Global * N_Global) / 256, 1, 1);
    dim3 BlockDim(WARP_SIZE, 1, 1);
    SplitK_Reduction<<<GridDim, BlockDim, 0, stream>>>(C, Reduction_Workspace, M_Global, N_Global, Split_K);
    return cudaGetLastError();
}

static int BankID_Minimum(std::vector<unsigned int> ItemsInBank[])
{
    int ID           = 0;
    int MinItemCount = ItemsInBank[0].size();
    for (int i = 1; i < 32; i++) {
        if (ItemsInBank[i].size() < MinItemCount) {
            ID           = i;
            MinItemCount = ItemsInBank[i].size();
        }
    }
    return ID;
}

static int BankID_Maximum(std::vector<unsigned int> ItemsInBank[])
{
    int ID           = 0;
    int MaxItemCount = ItemsInBank[0].size();
    for (int i = 1; i < 32; i++) {
        if (ItemsInBank[i].size() > MaxItemCount) {
            ID           = i;
            MaxItemCount = ItemsInBank[i].size();
        }
    }
    return ID;
}

/*
return: Number of Element in array TileOffsets
Note: TileOffsets[return-1] = NNZ / SparseKernelConfig::VECTOR_SIZE    (SparseKernelConfig::VECTOR_SIZE = 4)
*/
// template<typename TilingConfig, typename SparseKernelConfig>
__host__ int InitSparseMatrixA_API(half*      A_h,
                                   int        M,
                                   int        N,
                                   int        K,
                                   uint32_t** Compressed_A,  // CPU PTR
                                   int**      TileOffsets)        // CPU_PTR
{
    // Unified Sparse Fornat for different N, in our kernel, TILE_M=128 or 256
    const int TILE_M                       = 128;
    const int VECTOR_SIZE                  = 4;
    const int PADDING_SIZE_FOR_TILEOFFSETS = 2;
#ifdef DEBUG_MODE
    printf("Weight Shuffle is Enabled\n");
#endif
    float ZERO_THRESHOLD = 0.0;
    int   NumRow_offsets = M / TILE_M;
    int   NumCol_offsets = K / TILE_K;
    //
    int NNZ_Original = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            if (fabs(__half2float(A_h[i * K + j])) > ZERO_THRESHOLD)
                NNZ_Original++;
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ=%d, Pruning Ratio=%.2f\n",
           M,
           K,
           NNZ_Original,
           1.0f - static_cast<float>(NNZ_Original) / (M * K));
#endif
    //
    int  NNZ_AfterPadding   = 0;
    int* PaddingForEachTile = (int*)malloc(NumRow_offsets * NumCol_offsets * sizeof(int));
    if (!PaddingForEachTile) {
        printf("Error in InitSparseMatrixA line %d :malloc Error\n", __LINE__);
        exit(-1);
    }
    for (int i = 0; i < M / TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR = A_h + (i * TILE_M) * K + (j * TILE_K);
            int   TileNZCount    = 0;
            for (int m = 0; m < TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD)
                        TileNZCount++;
                }
            }
            int NumPadding                           = (VECTOR_SIZE - (TileNZCount % VECTOR_SIZE)) % VECTOR_SIZE;
            PaddingForEachTile[i * (K / TILE_K) + j] = NumPadding;
            TileNZCount += NumPadding;
            NNZ_AfterPadding += TileNZCount;
        }
    }
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ_AfterPadding=%d, PruningRatio_AfterPadding=%.2f\n",
           M,
           K,
           NNZ_AfterPadding,
           1.0f - static_cast<float>(NNZ_AfterPadding) / (M * K));
#endif
    //
    *Compressed_A = (uint32_t*)malloc(NNZ_AfterPadding * sizeof(uint32_t));
    *TileOffsets  = (int*)malloc((NumRow_offsets * NumCol_offsets + PADDING_SIZE_FOR_TILEOFFSETS) * sizeof(int));
    if (*Compressed_A == NULL || *TileOffsets == NULL) {
        printf("InitSparseMatrixA: Error in malloc memory from host memory!\n");
        exit(-1);
    }
    // Generating compressed format for A Matrix
    assert(M % TILE_M == 0 && K % TILE_K == 0);
    int       TotalNZCount = 0;
    uint32_t* Ptr_SubArray = *Compressed_A;
    for (int i = 0; i < M / TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half*        CurrentTilePTR    = A_h + (i * TILE_M) * K + (j * TILE_K);
            int          TileNZCount       = 0;
            int          remainingPaddings = PaddingForEachTile[i * (K / TILE_K) + j];
            unsigned int Item              = 0;
            // Processing each tile
            std::vector<unsigned int> ItemsInBank[32];
            int                       ZeroPositionForBank[32];
            for (int k = 0; k < 32; k++)
                ZeroPositionForBank[k] = -1;
            //
            // printf("Starting Processing Tile i:%d j:%d...\n", i, j);
            for (int m = 0; m < TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    // Row permutation for bank-conflict-free shared memory layout
                    int      row            = m;
                    int      col            = n;
                    uint32_t mask           = (row % 8) << 3;
                    int      col_permutated = col ^ mask;
                    int      bank_smem      = (col_permutated / 2) % 32;
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD) {
                        half* half_ptr   = reinterpret_cast<half*>(&Item);
                        *half_ptr        = value;
                        short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);
                        *short_ptr       = static_cast<short>(row * TILE_K + col_permutated);
                        ItemsInBank[bank_smem].push_back(Item);
                        //
                        TileNZCount++;
                    }
                    else {
                        if (ZeroPositionForBank[bank_smem] == -1)
                            ZeroPositionForBank[bank_smem] = row * TILE_K + col_permutated;
                    }
                }
            }
            //
            // printf("Starting Weight Padding...\n");
            for (int k = 0; k < remainingPaddings; k++) {
                int BankID = BankID_Minimum(ItemsInBank);
                assert(BankID >= 0 && BankID < 32);
                int ZeroPosition = ZeroPositionForBank[BankID];
                assert(ZeroPosition != -1);
                //
                half* half_ptr   = reinterpret_cast<half*>(&Item);
                *half_ptr        = __float2half_rn(0.0f);
                short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);
                *short_ptr       = static_cast<short>(ZeroPosition);
                ItemsInBank[BankID].push_back(Item);
                //
                TileNZCount++;
            }
            /*
            if(i==0 && j==0)
            {
              printf("For tile i:%d j:%d\n",i,j);
              for(int h=0; h<32; h++)
                printf("%ld ", ItemsInBank[h].size());
              printf("\n");
            }
            */
            //
            // printf("Starting Weight Shuffle...\n");
            std::vector<unsigned int> MainPart[32];
            std::vector<unsigned int> TailPart[32];
            int                       TileVectorCount = TileNZCount / VECTOR_SIZE;
            assert(TileNZCount % VECTOR_SIZE == 0);
            int Repeat_Vector   = TileVectorCount / WARP_SIZE;
            int Remained_Vector = TileVectorCount % WARP_SIZE;
            // Filing the TailPart
            for (int v = 0; v < VECTOR_SIZE; v++) {
                for (int b = 0; b < Remained_Vector; b++) {
                    int BankID = BankID_Maximum(ItemsInBank);
                    Item       = ItemsInBank[BankID].back();
                    ItemsInBank[BankID].pop_back();
                    TailPart[b].push_back(Item);
                }
            }
            // Filing the MainPart
            // printf("Starting Filing the MainPart...\n");
            for (int r = 0; r < Repeat_Vector; r++) {
                for (int v = 0; v < VECTOR_SIZE; v++) {
                    for (int b = 0; b < WARP_SIZE; b++) {
                        int BankID = BankID_Maximum(ItemsInBank);
                        Item       = ItemsInBank[BankID].back();
                        ItemsInBank[BankID].pop_back();
                        MainPart[b].push_back(Item);
                    }
                }
            }
            // Writing to the Sub-Array
            // printf("Starting Writing to the Sub-Array...\n");
            for (int r = 0; r < Repeat_Vector; r++) {
                for (int v = 0; v < VECTOR_SIZE; v++) {
                    for (int b = 0; b < 32; b++) {
                        Item = MainPart[b].back();
                        MainPart[b].pop_back();
                        int V_Size                                     = VECTOR_SIZE;
                        Ptr_SubArray[r * V_Size * 32 + b * V_Size + v] = Item;
                    }
                }
            }
            Ptr_SubArray += Repeat_Vector * VECTOR_SIZE * WARP_SIZE;
            for (int v = 0; v < VECTOR_SIZE; v++) {
                for (int b = 0; b < Remained_Vector; b++) {
                    Item = TailPart[b].back();
                    TailPart[b].pop_back();
                    Ptr_SubArray[b * VECTOR_SIZE + v] = Item;
                }
            }
            Ptr_SubArray += VECTOR_SIZE * Remained_Vector;
            //
            TotalNZCount += TileNZCount;
            (*TileOffsets)[i * K / TILE_K + j + 1] = TotalNZCount / VECTOR_SIZE;
        }
    }
    //
    assert(TotalNZCount == NNZ_AfterPadding);
    (*TileOffsets)[0] = 0;
    (*TileOffsets)[(M / TILE_M) * (K / TILE_K) + 1] =
        TotalNZCount / VECTOR_SIZE;  // #define PADDING_SIZE_FOR_TILEOFFSETS 2  // (N+1 offsets) + 1 padding // adding
                                     // an empty tile at last
    //
    return (M / TILE_M) * (K / TILE_K) + 2;  // number of Elements in array TileOffsets
}

// A_h is host memory pointer, Compressed_A and TileOffsets are device memory pointers
__host__ int InitSparseMatrixA_API_NoReorder(half*      A_h,
                                             int        M,
                                             int        N,
                                             int        K,
                                             uint32_t** Compressed_A,  // CPU PTR
                                             int**      TileOffsets)        // CPU_PTR
{
    // Unified Sparse Fornat for different N, in our kernel, TILE_M=128 or 256
    const int TILE_M                       = 128;
    const int VECTOR_SIZE                  = 4;
    const int PADDING_SIZE_FOR_TILEOFFSETS = 2;
#ifdef DEBUG_MODE
    printf("Weight Shuffle is NOT Enabled\n");
#endif
    float ZERO_THRESHOLD = 0.0;
    int   NumRow_offsets = M / TILE_M;
    int   NumCol_offsets = K / TILE_K;
    //
    int NNZ_Original = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            if (fabs(__half2float(A_h[i * K + j])) > ZERO_THRESHOLD)
                NNZ_Original++;
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ=%d, Pruning Ratio=%.2f\n",
           M,
           K,
           NNZ_Original,
           1.0f - static_cast<float>(NNZ_Original) / (M * K));
#endif
    //
    int  NNZ_AfterPadding   = 0;
    int* PaddingForEachTile = (int*)malloc(NumRow_offsets * NumCol_offsets * sizeof(int));
    if (!PaddingForEachTile) {
        printf("Error in InitSparseMatrixA line %d :malloc Error\n", __LINE__);
        exit(-1);
    }
    for (int i = 0; i < M / TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR = A_h + (i * TILE_M) * K + (j * TILE_K);
            int   TileNZCount    = 0;
            for (int m = 0; m < TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD)
                        TileNZCount++;
                }
            }
            int NumPadding                           = (VECTOR_SIZE - (TileNZCount % VECTOR_SIZE)) % VECTOR_SIZE;
            PaddingForEachTile[i * (K / TILE_K) + j] = NumPadding;
            TileNZCount += NumPadding;
            NNZ_AfterPadding += TileNZCount;
        }
    }
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ_AfterPadding=%d, PruningRatio_AfterPadding=%.2f\n",
           M,
           K,
           NNZ_AfterPadding,
           1.0f - static_cast<float>(NNZ_AfterPadding) / (M * K));
#endif
    //
    *Compressed_A = (uint32_t*)malloc(NNZ_AfterPadding * sizeof(uint32_t));
    *TileOffsets  = (int*)malloc((NumRow_offsets * NumCol_offsets + PADDING_SIZE_FOR_TILEOFFSETS) * sizeof(int));
    if (*Compressed_A == NULL || *TileOffsets == NULL) {
        printf("InitSparseMatrixA: Error in malloc memory from host memory!\n");
        exit(-1);
    }
    // Generating compressed format for A Matrix
    assert(M % TILE_M == 0 && K % TILE_K == 0);
    int TotalNZCount = 0;
    for (int i = 0; i < M / TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR    = A_h + (i * TILE_M) * K + (j * TILE_K);
            int   TileNZCount       = 0;
            int   remainingPaddings = PaddingForEachTile[i * (K / TILE_K) + j];
            for (int m = 0; m < TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD) {
                        half* half_ptr   = reinterpret_cast<half*>(*Compressed_A + TotalNZCount);
                        *half_ptr        = value;
                        short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);
                        // Row permutation for bank-conflict-free shared memory layout
                        int      row            = m;
                        int      col            = n;
                        uint32_t mask           = (row % 8) << 3;
                        int      col_permutated = col ^ mask;
                        *short_ptr              = static_cast<short>(row * TILE_K + col_permutated);
                        //
                        TileNZCount++;
                        TotalNZCount++;
                    }
                    else {
                        if (remainingPaddings > 0) {
                            remainingPaddings--;
                            half* half_ptr   = reinterpret_cast<half*>(*Compressed_A + TotalNZCount);
                            *half_ptr        = value;  // zero
                            short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);
                            // Row permutation for bank-conflict-free shared memory layout
                            int      row            = m;
                            int      col            = n;
                            uint32_t mask           = (row % 8) << 3;
                            int      col_permutated = col ^ mask;
                            *short_ptr              = static_cast<short>(row * TILE_K + col_permutated);
                            //
                            TileNZCount++;
                            TotalNZCount++;
                        }
                    }
                }
            }
            //
            assert(TileNZCount % VECTOR_SIZE == 0);
            (*TileOffsets)[i * K / TILE_K + j + 1] = TotalNZCount / VECTOR_SIZE;
        }
    }
    assert(TotalNZCount == NNZ_AfterPadding);
    (*TileOffsets)[0] = 0;
    (*TileOffsets)[(M / TILE_M) * (K / TILE_K) + 1] =
        TotalNZCount / VECTOR_SIZE;  // #define PADDING_SIZE_FOR_TILEOFFSETS 2  // (N+1 offsets) + 1 padding // adding
                                     // an empty tile at last
    //

    //
    return (M / TILE_M) * (K / TILE_K) + 2;  // number of Elements in array TileOffsets
}

/*
input:    char* DenseMatrixFileName
          int   M
          int   N                   // N is used by void InitSparseMatrixA_API()
          int   K
          char* NZWeightsFileName
          char* TileOffsetsFileName
          char* OutputSizesFileName // NNZ -> NumOffsets
*/
extern "C" void GenSparseMatrixBinFile(char* DenseMatrixFileName,
                                       int   M,
                                       int   K,
                                       char* NZWeightsFileName,
                                       char* TileOffsetsFileName,
                                       char* OutputSizesFileName)
{
    std::vector<half> host_array(M * K);
    std::ifstream     in(DenseMatrixFileName, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        printf("file %s cannot be opened, loadDataArrayFromBin fails. \n", DenseMatrixFileName);
        exit(-1);
    }
    size_t loaded_data_size = sizeof(half) * M * K;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);
#ifdef DEBUG_MODE
    printf("Read %ld bytes from %s.\n", loaded_data_size, DenseMatrixFileName);
#endif
    in.read((char*)host_array.data(), loaded_data_size);
    size_t in_get_size = in.gcount();
    if (in_get_size != loaded_data_size) {
        printf("file %s only has %ld, but request %ld, loading DenseMatrix fails! \n",
               DenseMatrixFileName,
               in_get_size,
               loaded_data_size);
        exit(-1);
    }
    in.close();
    // Step 2: Dense to Sparse Transformation
    unsigned int* NZWeights_CPU   = nullptr;
    int*          TileOffsets_CPU = nullptr;
    int           NumOffsets      = InitSparseMatrixA_API(host_array.data(), M, 0, K, &NZWeights_CPU, &TileOffsets_CPU);
    int           NNZ             = TileOffsets_CPU[NumOffsets - 1] * 4;  // VectorSize = 4
    // Step 3: Write to FILE(OutputSizesFileName)
    //         Write to FILE(NZWeightsFileName), FILE(TileOffsetsFileName)
    std::ofstream out_SizesFile(OutputSizesFileName, std::ios::out | std::ios::binary);
    std::ofstream out_NZWeightsFile(NZWeightsFileName, std::ios::out | std::ios::binary);
    std::ofstream out_TileOffsetsFile(TileOffsetsFileName, std::ios::out | std::ios::binary);
    if (!out_SizesFile.is_open() || !out_NZWeightsFile.is_open() || !out_TileOffsetsFile.is_open()) {
        printf("GenSparseMatrixBinFile() ERROR: file %s, %s, or %s cannot be opened or creaetd. \n",
               OutputSizesFileName,
               NZWeightsFileName,
               TileOffsetsFileName);
        exit(-1);
    }
    //
    // out_SizesFile << NNZ << NumOffsets;
    out_SizesFile.write((char*)&NNZ, sizeof(int));
    out_SizesFile.write((char*)&NumOffsets, sizeof(int));
    out_SizesFile.close();
    out_NZWeightsFile.write((char*)NZWeights_CPU, sizeof(uint32_t) * NNZ);
    out_NZWeightsFile.close();
    out_TileOffsetsFile.write((char*)TileOffsets_CPU, sizeof(int) * NumOffsets);
    out_TileOffsetsFile.close();
}