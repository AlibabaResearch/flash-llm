# Flash-LLM

Flash-LLM is an efficient GPU library for cost-effective and highly-efficient
large generative model inference. It provides efficient GPU MatMul kernels with
unstructured sparsity by leveraging Tensor Cores. With Flash-LLM, the pruned LLM
models can be deployed onto GPUs with less memory consumption and can execute
efficiently.

Currently, the code has been evaluated on NVIDIA A100 GPUs.

## Build Flash-LLM

#### 1. Prepare the Docker
Note: please specify PATH_LOCAL and PATH_DOCKER according to your preference.

```sh
sudo docker run -it —gpus all \
—name FlashLLM \
—privileged \
-v LOCAL_PATH:DOCKER_PATH \
nvcr.io/nvidia/pytorch:22.07-py3 bash
```

#### 2. Submodule Configuration

```sh
git clone https://github.com/AlibabaResearch/flash-llm.git
cd Flash-LLM
git submodule update --init --recursive
source Init_FlashLLM.sh
cd $FlashLLM_HOME/third_party/FasterTransformer && git am ../ft.patch
cd $FlashLLM_HOME/third_party/sputnik && git am ../sputnik.patch
```

#### 3. Building
The libSpMM_API.so and SpMM_API.cuh will be available for easy integration after:
```sh
cd $FlashLLM_HOME/build && make -j
```


## LLM Inference Example

#### 1. Building Faster-Transformer with Flash-LLM integration
```sh
cd $FlashLLM_HOME/third_party/FasterTransformer/
mkdir -p build && cd build
pip install -r ../examples/pytorch/gpt/requirement.txt
cmake -DSM=80 -DCMAKE_BUILD_TYPE=Release -DBUILD_MULTI_GPU=ON -DFLASH_LLM=ON ..
make -j
```
Note: if you want run standard Faster-Transformer (cuBLAS will be used for all MatMuls), use the following cmake command instead:
```sh
cmake -DSM=80 -DCMAKE_BUILD_TYPE=Release -DBUILD_MULTI_GPU=ON -DFLASH_LLM=OFF ..
make -j
```

#### 2. Downloading & Converting OPT models

Downloading the OPT model checkpoint (taking OPT-30B as an example) :
```sh
cd $FlashLLM_HOME/end2end_inference/models
git clone https://huggingface.co/facebook/opt-30b && opt-30b
apt install git-lfs
git lfs pull --include="pytorch_model*"
```
Converting from PyTorch format to Faster-Transformer format, here **-i_g** means the number of GPUs you will use for inference while -p means the number of CPU threads you use for this model format transformation. This script will load pytorch model from -i PATH and output the generated model to -o PATH.
```sh
cd $FlashLLM_HOME/end2end_inference/ft_tools
python huggingface_opt_convert_Phase1.py \
      -i $FlashLLM_HOME/end2end_inference/models/opt-30b \
      -o $FlashLLM_HOME/end2end_inference/models/opt-30b/c-model \
      -i_g 1 \
      -weight_data_type fp16 \
      -p 64
```

Converting from Faster-Transformer format to Flash-LLM format. Note that here **-i_g** means the number of GPUs you will use for inference. We fake 80% sparsity in this script for demonstration purpose, which can be disabled if you already have your OPT models pruned. Besides, hand-tuned SplitK is used for different input shapes.
```sh
python huggingface_opt_convert_Phase2.py \
      -m $FlashLLM_HOME/end2end_inference/models/opt-30b/c-model \
      -l $FlashLLM_HOME/build/libSpMM_API.so \
      -i_g 1 \
      -p 64
```

#### 3. Configuration
First, you need to modify the config file for the inference task in path:
*$FlashLLM_HOME/third_party/FasterTransformer/examples/cpp/multi_gpu_gpt/gpt_config.ini*
The most important thing is to change the model_dir according to your absolute file path.
The model_name should also be changed if you are running other OPT models.
Besides, the tensor_para_size should be the same as the **-i_g** you used in step 2.

#### 4. Running Inference
Note that **-n** here means the number of GPUs you use for inference, which should be the same as **-ig** in step 2.
```sh
cd $FlashLLM_HOME/third_party/FasterTransformer/build
mpirun -n 1 --allow-run-as-root ./bin/multi_gpu_gpt_example
```

#### 5. DeepSpeed baseline
```sh
cd $FlashLLM_HOME/end2end_inference/ds_scripts
pip install -r requirements.txt
deepspeed --num_gpus 1 inference-test.py --ds_inference --greedy --use_meta_tensor --use_kernel --name facebook/opt-30b --batch_size 8 --max_new_tokens 512 --max_tokens 576
```


## Kernel Benchmarking

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


## Citation
Paper stay tuned.