from funasr_torch import Paraformer_dashscope as Paraformer


# model_dir = "/home/haoneng.lhn/FunASR/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
# model_dir = "damo_chunk/speech_dashscope_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
model_dir = "amp_int8_chunk/libtorch"
model_dir = "amp_int8_chunk/libtorch_fp16"
model_dir = "amp_int8_chunk/bladedisc_fp16"
if 'blade' in model_dir:
    import torch_blade

# model = Paraformer(model_dir, batch_size=1, quantize=False)  # cpu
model = Paraformer(model_dir, batch_size=4, device_id=0, quantize=False)  # gpu

# wav_path = "/home/haoneng.lhn/FunASR/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav"
wav_path = "damo_chunk/speech_dashscope_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav"

result = model(wav_path)
print(result)

import time
for i in range(100):
    tic = time.time()
    model(wav_path)
    print(time.time() - tic)
