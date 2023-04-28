import json
import numpy as np
import torch
from torch.utils.dlpack import from_dlpack

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
        self.device = 'cpu'
        # self.device = 'cuda'

        out_config = pb_utils.get_output_config_by_name(model_config, 'text')
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

        """
        speech = []
        speech_len = []
        responses = []
        for request in requests:
            _speech = pb_utils.get_input_tensor_by_name(request, 'speech')
            _speech_len = pb_utils.get_input_tensor_by_name(request, 'speech_lengths')
            speech.append(_speech.as_numpy().squeeze(0))
            speech_len.append(int(_speech_len.as_numpy().squeeze()))
        max_feat_len = max(speech_len)
        for i, _speech in enumerate(speech):
            paddings = ((0, max_feat_len - speech_len[i]), (0, 0))
            speech[i] = np.pad(_speech, paddings, 'constant', constant_values=0)
        feature = torch.from_numpy(np.array(speech))
        feature_len = torch.tensor(speech_len, dtype=torch.int32).to(self.device)
        """
        """
        speech = []
        speech_len = []
        responses = []
        for request in requests:
            _speech = pb_utils.get_input_tensor_by_name(request, 'speech')
            _speech_len = pb_utils.get_input_tensor_by_name(request, 'speech_lengths')
            _speech = from_dlpack(_speech.to_dlpack())
            _speech_len = from_dlpack(_speech_len.to_dlpack())
            speech.append(_speech)
            speech_len.append(int(_speech_len))
        shape = (len(speech), max(speech_len), speech[0].shape[-1])
        feature = torch.zeros(shape, dtype=speech[0].dtype, device=self.device)
        for i, _speech in enumerate(speech):
            feature[i,:speech_len[i]] = _speech
        feature_len = torch.tensor(speech_len, dtype=torch.int32).to(self.device)
        """

        """
        results = self.model(feature, feature_len)

        for res in results:
            text = np.array(res['preds'][0])
            out = pb_utils.Tensor('text', text.astype(self.out_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out])
            responses.append(inference_response)
        return responses
        """

        # """
        responses = []
        for _ in requests:
            text = np.array("None")
            out = pb_utils.Tensor('text', text.astype(self.out_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out])
            responses.append(inference_response)
        return responses
        # """
