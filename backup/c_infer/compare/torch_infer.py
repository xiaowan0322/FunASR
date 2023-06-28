import sys
import time
import torch
import torch_blade


def load_data_lst(start_i, end_i):
    data_lst = list()
    for i in range(start_i, end_i):
        data_file = 'test_data/test_{}.pth'.format(i)
        test_data = torch.load(data_file)
        data_lst.append(test_data)
    return data_lst

def infer(module, data_lst, warmup_num=0):
    num = len(data_lst) if warmup_num == 0 else warmup_num
    torch.cuda.synchronize()
    tic = time.time()
    with torch.no_grad():
        for i in range(num):
            # t0 = torch.zeros(data_lst[i][0].shape, dtype=torch.float32)
            out = module(*data_lst[i])
            torch.cuda.synchronize()
    cost_time = time.time() - tic
    return cost_time * 1000.0


if __name__ == '__main__':
    model_file = sys.argv[1]
    start_i, end_i = int(sys.argv[2]), int(sys.argv[3])
    module = torch.jit.load(model_file, map_location=torch.device('cuda'))
    module.eval()
    data_lst = load_data_lst(start_i, end_i)

    t0 = infer(module, data_lst, warmup_num=50)
    t1 = infer(module, data_lst)
    print('t1: {:.2f}'.format(t1))
    t2 = infer(module, data_lst)
    print('t2: {:.2f}'.format(t2))
