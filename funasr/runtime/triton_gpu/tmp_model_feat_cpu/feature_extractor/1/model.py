import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack
import torch
import numpy as np
import json
from typing import Any, Dict, Iterable, List, NamedTuple, Set, Tuple, Union
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
        self.max_batch_size = max(model_config["max_batch_size"], 1)
        self.device = "cuda"

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "speech")
        # Convert Triton types to numpy types
        output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        if output0_dtype == np.float32:
            self.output0_dtype = torch.float32
        else:
            self.output0_dtype = torch.float16

        # Get OUTPUT1 configuration
        output1_config = pb_utils.get_output_config_by_name(
            model_config, "speech_lengths")
        # Convert Triton types to numpy types
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config['data_type'])

        model_path = str(model_config['parameters']['model_path']['string_value'])
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
        total_waves = []
        responses = []
        for request in requests:
            input0 = pb_utils.get_input_tensor_by_name(request, "wav")
            cur_b_wav = input0.as_numpy().squeeze() * (1 << 15) # b x -1
            total_waves.append(cur_b_wav)

        features, feats_len = self.model.extract_feat(total_waves)

        for i in range(features.shape[0]):
            speech = features[i:i+1][:int(feats_len[i])]
            speech_lengths = feats_len[i].unsqueeze(0).unsqueeze(0)

            speech, speech_lengths = speech.cpu(), speech_lengths.cpu()
            out0 = pb_utils.Tensor.from_dlpack("speech", to_dlpack(speech))
            out1 = pb_utils.Tensor.from_dlpack("speech_lengths",
                                               to_dlpack(speech_lengths))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out0, out1])
            responses.append(inference_response)
        return responses
