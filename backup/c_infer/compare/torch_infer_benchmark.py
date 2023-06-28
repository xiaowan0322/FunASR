import sys
import time
import torch
import torch_blade


def inference(model_file, data_file, batch_size=1):
    module = torch.jit.load(model_file, map_location=torch.device('cuda'))
    test_data = torch.load(data_file)

    module.eval()
    def run():
        with torch.no_grad():
            out = module(*test_data)
            torch.cuda.synchronize()
        return out

    # warmup
    for i in range(20):
        out = run()

    tic = time.time()
    for i in range(100):
        out = run()
    cost_time = time.time() - tic
    print('avg_time {}'.format(cost_time / 100))


if __name__ == '__main__':
    model_file = sys.argv[1]
    data_file = sys.argv[2]
    batch_size = int(sys.argv[3])
    inference(model_file, data_file, batch_size)
