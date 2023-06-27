import torch
import torch_blade
import pdb

import os
# os.environ["TORCH_BLADE_DEBUG_LOG"] = "true"
# os.environ["TORCH_BLADE_MHLO_DEBUG_LOG"] = "true"
# os.environ["PYTORCH_JIT_LOG_LEVEL"] = ">>>mhlo_conversion"
# os.environ["TORCH_BLADE_DEBUG_ENABLE_ERROR_FALLBACK"] = "true"
# os.environ["TORCH_DISC_ENABLE_REPLAY_ON_CLUSTER"] = "true"
# os.environ["TORCH_BLADE_DEBUG_ENABLE_ERROR_FALLBACK"] = "true"


inputs = torch.load('data/test_data_1x93x560.pt')
module = torch.jit.load('z_models/sub_encoder_fixfp16.pt')

# inputs = torch.load('data/dummy_decoder_data.pth')
# module = torch.jit.load('funasr_export/speech_dashscope_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/decoder.torchscripts')

module = module.eval()
inputs = tuple([tuple([j.cuda() for j in i]) if isinstance(i, tuple) else i.cuda() for i in inputs])

# pdb.set_trace()
with torch.no_grad():
    module(*inputs)

fp16 = True
if fp16:
    torch_config = torch_blade.config.Config()
    torch_config.enable_fp16 = True
    with torch.no_grad():
        with torch_config:
            blade_module = torch_blade.optimize(module, allow_tracing=True, model_inputs=inputs)
else:
    with torch.no_grad():
        blade_module = torch_blade.optimize(module, allow_tracing=True, model_inputs=inputs)

# run the optimized module
with torch.no_grad():
    blade_module(*inputs)

torch.jit.save(blade_module, 'blade.fp16.pt' if fp16 else 'blade.pt')
