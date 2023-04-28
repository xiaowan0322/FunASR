import json
import numpy as np
import torch

import triton_python_backend_utils as pb_utils
from funasr_torch import Paraformer


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        self.model_config = model_config = json.loads(args['model_config'])
        self.max_batch_size = max(model_config['max_batch_size'], 1)
        self.device = 'cuda'

        out_config = pb_utils.get_output_config_by_name(model_config, 'TEXT')
        self.out_dtype = pb_utils.triton_string_to_numpy(out_config['data_type'])

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model_path = str(model_config['parameters']['model_path']['string_value'])
        if 'blade' in model_path:
            import torch_blade
        self.model = Paraformer(model_path)

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        waveform_list = []
        responses = []
        for request in requests:
            input_wav = pb_utils.get_input_tensor_by_name(request, 'WAV')
            cur_b_wav = input_wav.as_numpy().squeeze()
            cur_b_wav = cur_b_wav * (1 << 15)  # b x -1
            waveform_list.append(cur_b_wav)

        results = self.model(waveform_list)

        for res in results:
            text = np.array(res['preds'][0])
            out = pb_utils.Tensor('TEXT', text.astype(self.out_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out])
            responses.append(inference_response)
        return responses
