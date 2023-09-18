# Kernel Benchmarking

#### 1. Installing Sputnik
```sh
cd $FlashLLM_HOME/third_party/sputnik
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=OFF -DBUILD_BENCHMARK=OFF -DCUDA_ARCHS=”80”
make -j12
```

#### 2. Installing cuSPARSELT
Following this [link](https://developer.nvidia.com/cusparselt-downloads).

#### 3. Building spmm_test
```sh
cd $FlashLLM_HOME/kernel_benchmark
source test_env
make
```

#### 4. Benchmarking
```sh
./benchmark.sh 
```

