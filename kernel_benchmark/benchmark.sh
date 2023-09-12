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


M=(21504  7168   28672  7168   27648  9216   36864  9216   36864  12288  49152  12288)
K=(7168   7168   7168   28672  9216   9216   9216   36864  12288  12288  12288  49152)
SplitK=(5      7      7      7      2      6      3      6      3      9      9     9)
N=(8 16 32 64)
Sparsity=(70 80 90)


for BS in ${N[@]} 
do
    #echo "BS=${BS}"
    for ((i=0;i<${#M[@]};i++)) 
    do
        #echo "Processing Shape ${i}..."
        for S in ${Sparsity[@]} 
        do
            #echo "Sparsity = $S"
                ./spmm_test ${M[i]} ${K[i]} ${BS} ${S} ${SplitK[i]}
        done    
    done
done

# ./spmm_test 36864 9216 32 70 3