import torch
import copy

import time
import pdb


class MyConv1d(torch.nn.Module):
    def __init__(self):
        super(MyConv1d, self).__init__()
        # (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
        kernel_size = 11
        fsmn_shift = 0
        # padding
        left_padding = (kernel_size - 1) // 2
        if fsmn_shift > 0:
            left_padding = left_padding + fsmn_shift
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = torch.nn.ConstantPad1d((left_padding, right_padding), 0.0)
        self.conv = torch.nn.Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
        # self.conv = torch.nn.Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=1, bias=False)
        self.conv.half()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.pad_fn(x)
        # x = self.conv(x)
        x = (self.conv(x.half())).float()
        x = self.relu(x)
        return x


if __name__ == '__main__':
    torch.manual_seed(0)
    model = MyConv1d()
    model = model.cuda()
    input_tensor = torch.randn(64, 512, 93)
    input_tensor = input_tensor.cuda()
    print(input_tensor[0, 0, :10])
    output = model(input_tensor)
    print(output.shape)
    print(output[0, 0, :10])
    pdb.set_trace()

    model_script = torch.jit.trace(model, input_tensor)
    # model_script = torch.jit.script(model)
    model_script.save('conv1d.pt')

    print('finish')
