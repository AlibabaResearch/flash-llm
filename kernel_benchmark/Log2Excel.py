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

# Note: please enable the test of all kernels in spmm_test.cu
import pandas as pd

lines = []
# open the log file for reading
with open("log.txt", "r") as f:
    lines = f.readlines()

# find the line that contains performance data
TimeList = []
for line in lines:
    if "Time/ms:" in line:
        time = float(line.split(":")[1].split(" ")[1])
        TimeList.append(time)

# Number of Columns
N_COL = 8
data = []
assert( len(TimeList)%N_COL == 0 )
for i in range( len(TimeList)//N_COL ):
    row = []
    for j in range(N_COL):
        row.append( TimeList[i*N_COL+j] )
    data.append(row)

# create a pandas dataframe from the performance data
df = pd.DataFrame(data, columns=["CuSparse_C", "CuSparse_R", "Sputnik", "CuBlas_SIMT", "CuBlas_TC", "Flash-LLM", "Flash-LLM_NoReorder", "sparTA"])
# write the dataframe to an Excel file
df.to_excel("KernelPerformance.xlsx", index=False)