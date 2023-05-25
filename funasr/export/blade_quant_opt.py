import torch
import torch_blade
import os
import sys
import pdb


os.environ['TORCH_BLADE_DEBUG_LOG'] = 'true'
os.environ['TORCH_BLADE_MHLO_DEBUG_LOG'] = 'true'

# pdb.set_trace()
model_file = sys.argv[1]
data_file = sys.argv[2]

traced = torch.jit.load(model_file)
_inputs = torch.load(data_file)

_inputs = tuple([tuple([j.cuda() for j in i]) \
    if isinstance(i, tuple) else i.cuda() for i in _inputs])

if 'mask' in model_file and 'encoders' in model_file:
    _inputs = (_inputs[0], _inputs[1][0], _inputs[1][1])


# [OPT]
if sys.argv[3] == 'opt':
    pdll_files = '/workspace/Workspace/BladeDISC/pytorch_blade/tests/disc/pdl/pdll_files'
    disc_torch_pdll_files = [
        os.path.join(pdll_files, 'common/fake_quant.pdll'),
        # os.path.join(pdll_files, 'gpu/dequant_gemm_quant.pdll'),
        # os.path.join(pdll_files, 'gpu/dequant_gemm_quant_bias_quant.pdll'),
        os.path.join(pdll_files, 'gpu/dequant_gemm_quant_bias_f32_quant.pdll'),
    ]
    os.environ['DISC_TORCH_PDL_FILES'] = ','.join(disc_torch_pdll_files)

    # stitch
    # os.environ['DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL'] = 'true'
    # os.environ['DISC_ENABLE_STITCH'] = 'true'

    blade_config = torch_blade.config.Config()
    blade_config.enable_int8 = True
    blade_config.enable_fp16 = True
    blade_config.disc_cluster_max_iter_count = 200
    with torch.no_grad(), blade_config:
        # traced = dict(traced.named_children())['0']
        quant_model = torch_blade.optimize(traced, model_inputs=_inputs)

    # pdb.set_trace()
    with torch.no_grad():
        output = quant_model(*_inputs)
    print(output)

    torch.jit.save(quant_model, 'blade.int8.pt')

# [RUN]
if sys.argv[3] == 'run':
    with torch.no_grad():
        output = traced(*_inputs)
    print(output)

# [TIME]
if sys.argv[3] == 'time':
    import time
    for i in range(1000):
        with torch.no_grad():
            tic = time.time()
            output = traced(*_inputs)
            print(time.time() - tic)
