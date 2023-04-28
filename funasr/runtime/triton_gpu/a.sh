# docker build . -f Dockerfile/Dockerfile.server -t triton-paraformer:22.08
# docker run -it --name paraformer_triton_server --gpus all -v /root/work_data/Workspace/FunASR_xw/funasr/runtime/triton_gpu:/workspace/ --shm-size 1g --net host triton-paraformer:22.08 bash

# tritonserver --model-repository /workspace/model_repo_paraformer_large_offline --pinned-memory-pool-byte-size=512000000 --cuda-memory-pool-byte-size=0:1024000000
tritonserver --model-repository /workspace/$1 --pinned-memory-pool-byte-size=512000000 --cuda-memory-pool-byte-size=0:1024000000

# tritonserver --model-repository /workspace/$1 --backend-config=python,shm-region-prefix-name=prefix1

# docker ps -a | grep triton-paraformer | xargs -i docker stop {}




# pip3 install librosa==0.8.1 --index-url http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com --disable-pip-version-check
# pip3 install typeguard==2.13.3 --index-url http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com --disable-pip-version-check
# pip3 install kaldi-native-fbank==1.15 --index-url http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com --disable-pip-version-check
# pip3 install scripts/torch_blade-0.2.0+1.12.0.cu113-cp38-cp38-linux_x86_64.whl --index-url http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com --disable-pip-version-check



# ps -ef  | grep tritonserver | awk '{print "kill -9 " $2}' | sh
