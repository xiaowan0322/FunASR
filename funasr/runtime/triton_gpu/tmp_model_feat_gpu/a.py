import torch
import numpy as np


a1 = torch.tensor(np.zeros((8, 4), dtype=np.float32))
a2 = torch.tensor(np.zeros((11, 4), dtype=np.float32))
speech = [a1, a2]
speech_len = [int(torch.tensor(8, dtype=torch.int32)), int(torch.tensor(11, dtype=torch.int32))]
max_len = max(speech_len)
for i, _speech in enumerate(speech):
    pad = (0, 0, 0, max_len - speech_len[i])
    speech[i] = torch.nn.functional.pad(_speech, pad, mode='constant', value=0)

import pdb; pdb.set_trace()
feature = torch.tensor(speech, dtype=speech[0].dtype).to(self.device)
