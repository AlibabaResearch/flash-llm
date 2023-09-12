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

// used for the reduction of result matrix if Split-K is used
// Reduction_Workspace:     (M_Global, N_Global, Split_K),  column major
// C:                       (M_Global, N_Global),           column major
// Each thread deals with 8 output elements, each elements is the sum of Split_K elements
// Each Warp: 32 threads_per_warp * 8 half_per_threads -> 256 half_per_warp
// Each GPU: 108 SM -> 108 warp -> 108*256 = 27648
// GridSize = (M_Global*N_Global) / 256

#define ELEMENT_PER_THREADBLOCK 256

__global__ void SplitK_Reduction(half* C, half* Reduction_Workspace, int M_Global, int N_Global, int Split_K)
{
    // return;
    half* C_BasePTR_ThisBlock = C + ELEMENT_PER_THREADBLOCK * blockIdx.x;
    half* R_BasePTR_ThisBlock = Reduction_Workspace + ELEMENT_PER_THREADBLOCK * blockIdx.x;
    //
    float Results[HALF_PER_128B];
//
#pragma unroll
    for (int j = 0; j < HALF_PER_128B; j++)
        Results[j] = 0.0f;
    //
    for (int i = 0; i < Split_K; i++) {
#pragma unroll
        for (int j = 0; j < HALF_PER_128B; j++)
            Results[j] += __half2float(R_BasePTR_ThisBlock[threadIdx.x * HALF_PER_128B + j]);
        R_BasePTR_ThisBlock += M_Global * N_Global;
    }
#pragma unroll
    for (int j = 0; j < HALF_PER_128B; j++)
        C_BasePTR_ThisBlock[threadIdx.x * HALF_PER_128B + j] = __float2half_rn(Results[j]);
}
