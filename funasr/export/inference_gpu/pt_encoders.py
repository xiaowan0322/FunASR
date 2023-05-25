class Prof:
    def __init__(self):
        self._cudart = None
        cudart_lib_path = '/usr/local/cuda/lib64/libcudart.so'
        import os
        if os.path.exists(cudart_lib_path):
            import ctypes
            self._cudart = ctypes.CDLL(cudart_lib_path)
    def start(self):
        if self._cudart is None:
            return
        res = self._cudart.cudaProfilerStart()
        if res != 0:
            raise Exception('cudaProfilerStart returned {}'.format(res))
    def stop(self):
        if self._cudart is None:
            return
        res = self._cudart.cudaProfilerStop()
        if res != 0:
            raise Exception('cudaProfilerStop returned {}'.format(res))
prof = Prof()


import sys
import time
import torch


def recognize(model_file, data_file, batch_size=1, tf32=-1):
    if 'blade' in model_file:
        import torch_blade
    # """
    gpu = torch.device('cuda')
    module = torch.jit.load(model_file, map_location=gpu)
    # """
    # module = torch.jit.load(model_file)
    test_data = torch.load(data_file)
    # import pdb; pdb.set_trace()
    _istuple = lambda x: isinstance(x, tuple)
    _mod = lambda x, func: tuple([func(i) for i in x]) if _istuple(x) else func(x)
    _batch = lambda x: torch.cat([x] * batch_size)
    test_data = tuple([_mod(d, _batch) for d in test_data])
    _cuda = lambda x: x.cuda()
    test_data = tuple([_mod(d, _cuda) for d in test_data])

    """
    import os
    fp16 = float(os.environ.get('FP16', False))
    if fp16:
        test_data = (test_data[0] / fp16, test_data[1])
        if 'blade' not in model_file:
            _half = lambda x: x.half()
            test_data = tuple([_mod(d, _half) for d in test_data])
            module.half()
    """
    """
    _half = lambda x: x.half()
    test_data = tuple([_mod(d, _half) for d in test_data])
    module.half()
    """

    # if torch.cuda.is_available():
    #     test_data = tuple([i.cuda() for i in test_data])

    print(torch.backends.cuda.matmul.allow_tf32)
    print(torch.backends.cudnn.allow_tf32)
    if tf32 in [0, 1]:
        tf32 = tf32 != 0
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
    print(torch.backends.cuda.matmul.allow_tf32)
    print(torch.backends.cudnn.allow_tf32)

    """
    # inputs[0]       torch.Size([1, 93, 512])
    # inputs[1][0]    torch.Size([1, 93, 1])
    # inputs[1][1]    torch.Size([1, 1, 1, 93])
    length = 16
    # import pdb; pdb.set_trace()
    test_data = (
        test_data[0][:, :length, :],
        (test_data[1][0][:, :length, :], test_data[1][1][:, :, :, :length])
    )
    """
    # if 'mask' in model_file and 'encoders' in model_file:
    #     test_data = (test_data[0], test_data[1][0], test_data[1][1])

    with torch.no_grad():
        module.eval()
        def run():
            if isinstance(test_data, torch.Tensor):
                out = module(test_data)
            else:
                out = module(*test_data)
            torch.cuda.synchronize()
            return out

        # warmup
        for i in range(20):
            out = run()

        print(out[0])
        print(out[0].shape)
        import pdb; pdb.set_trace()

        prof.start()
        all_time = 0
        for i in range(100):
            tic = time.time()
            out = run()
            cost_time = time.time() - tic
            print(cost_time)
            all_time += cost_time
        print('avg_time {}'.format(all_time / 100))


model_file = sys.argv[1]
data_file = sys.argv[2]
batch_size = int(sys.argv[3])
tf32 = int(sys.argv[4])
recognize(model_file, data_file, batch_size, tf32)
