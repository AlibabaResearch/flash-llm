# Copyright 2023 The FLash-LLM Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#! /bin/bash
M=(21504  7168   28672  7168   27648  9216   36864  9216   36864  12288  49152  12288)
K=(7168   7168   7168   28672  9216   9216   9216   36864  12288  12288  12288  49152)
SplitK=(5      7      7      7      2      6      3      6      3      9      9     9)
N=(8 16 32 64)
Sparsity=(70 80 90)


echo "profiling Flash-LLM kernel..."
mkdir -p ProfilingSpMM
for ((i=0;i<${#M[@]};i++)) 
do
    echo "Processing Shape ${i}..."
    for BS in ${N[@]} 
    do
        echo "BS=${BS}"
        for S in ${Sparsity[@]} 
        do
            echo "Sparsity = $S"
            ncu -c 2 -f -o ProfilingSpMM/MM_SparseTCM${M[i]}K${K[i]}N${BS}S${S} --set full --kernel-name SpMM_Kernel \
                ./spmm_test ${M[i]} ${K[i]} ${BS} ${S} ${SplitK[i]}
        done    
    done
done

#echo "profiling cuBLAS kernel..."
#mkdir -p ProfilingcuBLAS
#for ((i=0;i<${#M[@]};i++)) 
#do
#    echo "Processing Shape ${i}..."
#    for BS in ${N[@]} 
#    do
#        echo "BS=${BS}"
#        ncu -c 1 -f -o ProfilingcuBLAS/cuBLASM${M[i]}K${K[i]}N${BS}S${S} --set full \
#            ./spmm_test ${M[i]} ${K[i]} ${BS} 70 ${SplitK[i]}
#    done
#done


#echo "profiling cuSPARSE kernel...""
#mkdir -p ProfilingcuSPARSE
#for ((i=0;i<${#M[@]};i++)) 
#do
#    echo "Processing Shape ${i}..."
#    for BS in ${N[@]} 
#    do
#        echo "BS=${BS}"
#        for S in ${Sparsity[@]} 
#        do
#            echo "Sparsity = $S"
#            ncu -f -o ProfilingcuSPARSE/cuSPARSE_M${M[i]}K${K[i]}N${BS}S${S} --set full --kernel-name regex:csrmm_alg2_kernel* \
#                ./spmm_test ${M[i]} ${K[i]} ${BS} ${S} ${SplitK[i]}
#        done    
#    done
#done


# Note: when profiling Sputnik, please disable the test of SparTA  as SparTA also uses Sputnik kernels
# by commenting out "#define USE_SPUTNIK" in spmm_test.cu

#echo "profiling Sputnik...""
#mkdir -p ProfilingSputnik
#for ((i=0;i<${#M[@]};i++)) 
#do
#    echo "Processing Shape ${i}..."
#    for BS in ${N[@]} 
#    do
#        echo "BS=${BS}"
#        for S in ${Sparsity[@]} 
#        do
#            echo "Sparsity = $S"
#            ncu -f -o ProfilingSputnik/Sputnik_M${M[i]}K${K[i]}N${BS}S${S} --set full --kernel-name Kernel \
#                ./spmm_test ${M[i]} ${K[i]} ${BS} ${S} ${SplitK[i]}
#        done    
#    done
#done


# Note: when profiling Sputnik, please disable the test of Sputnik
# by commenting out "#define USE_SPARTA" in spmm_test.cu

#echo "profiling SparTA..."
#mkdir -p ProfilingSparTA
#for ((i=0;i<${#M[@]};i++)) 
#do
#    echo "Processing Shape ${i}..."
#    for BS in ${N[@]} 
#    do
#        echo "BS=${BS}"
#        for S in ${Sparsity[@]} 
#        do
#            echo "Sparsity = $S"
#            ncu -f -o ProfilingSparTA/SparTA_M${M[i]}K${K[i]}N${BS}S${S} --set full --kernel-name regex:sm80_xmma* \
#                ./spmm_test ${M[i]} ${K[i]} ${BS} ${S} ${SplitK[i]}
#        done    
#    done
#done