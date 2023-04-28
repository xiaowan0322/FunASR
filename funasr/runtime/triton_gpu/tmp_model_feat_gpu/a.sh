# docker run -it --name client_test --net host --gpus all -v /root/work_data/Workspace/FunASR_xw/funasr/runtime/triton_gpu:/workspace/ soar97/triton-k2:22.12.1

# mkdir -p /root/fangjun/open-source/icefall-aishell/egs/aishell/ASR/download/aishell
# tar xf ./aishell-test-dev-manifests/data_aishell.tar.gz -C /root/fangjun/open-source/icefall-aishell/egs/aishell/ASR/download/aishell/

serveraddr=localhost
manifest_path=/workspace/aishell-test-dev-manifests/data/fbank/aishell_cuts_test.jsonl.gz
num_task=64
python3 decode_manifest_triton.py \
    --server-addr $serveraddr \
    --compute-cer \
    --model-name infer_pipeline \
    --num-tasks $num_task \
    --manifest-filename $manifest_path
