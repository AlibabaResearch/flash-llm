# Preparations

#### 1. Prepare the Docker
Note: please specify PATH_LOCAL and PATH_DOCKER according to your preference.

```sh
sudo docker run -it —gpus all \
—name FlashLLM \
—privileged \
-v $LOCAL_PATH:$DOCKER_PATH \
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
