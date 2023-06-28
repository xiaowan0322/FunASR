import sys
import time
import torch
import torch_blade


def inference(model_file):
    module = torch.jit.load(model_file, map_location=torch.device('cuda'))
    module.eval()

    def run(test_data):
        with torch.no_grad():
            out = module(*test_data)
            torch.cuda.synchronize()
        return out

    for i in range(7176):
        data_file = 'test_data/test_{}.pth'.format(i)
        test_data = torch.load(data_file)
        tic = time.time()
        out = run(test_data)
        cost_time = time.time() - tic
        print('{} {} {}'.format(i, cost_time * 1000, test_data[0].shape[1]))
    # print('avg_time {}'.format(cost_time / 100))


if __name__ == '__main__':
    model_file = sys.argv[1]
    inference(model_file)
