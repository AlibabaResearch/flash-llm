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
#ifndef TILINGCONFIG_H
#define TILINGCONFIG_H

//#define DEBUG_MODE 1
#define RELEASE_MODE 1

// Fixed Parameters
#define MMA_M 16
#define MMA_N 16
#define MMA_K 16
#define WARP_SIZE 32
// Unchangable
#define WARP_ROW_TENSORS 4
#define BLOCK_K_TENSORS 4
#define TILE_K (MMA_K * BLOCK_K_TENSORS)  // 64
// Parameters for copying A_TILE & B_TILE & C_TILE
#define COPY_UNIT_FP16_ROWS 8
#define COPY_UNIT_FP16_COLS 64
#define HALF_PER_128B 8           // LDS.128 -> 8 * FP16
#define REG_PER_C_TENSOR_16_16 8  // 8 for FP32 Accumulation; 4 for FP16 Accumulation

#define PADDING_SHARED_MEM_FOR_C 4  // Padding 8/2 float each column to eliminating bank-conflict in C fragments

template<int BLOCK_ROW_WARPS_, int BLOCK_COL_WARPS_, int WARP_COL_TENSORS_, int N8_ = 0>
struct TilingConfig {
    static constexpr int BLOCK_ROW_WARPS  = BLOCK_ROW_WARPS_;
    static constexpr int BLOCK_COL_WARPS  = BLOCK_COL_WARPS_;
    static constexpr int WARP_COL_TENSORS = WARP_COL_TENSORS_;
    // Sanity checks on the template arguments.
    // static_assert((BLOCK_ROW_WARPS * BLOCK_COL_WARPS) == 4,
    //               "The number of WARPS per threadblock must be 4.");
    // Derived Parameters
    static constexpr int TILE_M        = MMA_M * (WARP_ROW_TENSORS * BLOCK_ROW_WARPS);
    static constexpr int TILE_N        = MMA_N * (WARP_COL_TENSORS * BLOCK_COL_WARPS);
    static constexpr int BLOCK_WARPS   = BLOCK_ROW_WARPS * BLOCK_COL_WARPS;
    static constexpr int BLOCK_THREADS = BLOCK_WARPS * WARP_SIZE;
    // temporary implementation to support N=8
    static constexpr int N8      = N8_;
    static constexpr int TILE_N2 = N8 ? 8 : TILE_N;
};

template<int NUM_REG_FOR_SPARSE_KERNEL_ = 64>
struct SparseKernelConfig {
    static constexpr int NUM_REG_FOR_SPARSE_KERNEL    = NUM_REG_FOR_SPARSE_KERNEL_;
    static constexpr int VECTOR_SIZE                  = 4;
    static constexpr int PADDING_SIZE_FOR_TILEOFFSETS = 2;  // (N+1 offsets) + 1 padding
    // Sanity checks on the template arguments.
    // static_assert((BLOCK_ROW_WARPS * BLOCK_COL_WARPS) == 4,
    //               "The number of WARPS per threadblock must be 4.");
    // Derived Parameters
    // static constexpr int TILE_M = MMA_M * (WARP_ROW_TENSORS * BLOCK_ROW_WARPS);
};

#endif