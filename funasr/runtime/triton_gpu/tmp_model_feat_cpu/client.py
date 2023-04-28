from tritonclient.utils import np_to_triton_dtype
import tritonclient.grpc as grpcclient
import librosa
import sys
import numpy as np


model_name = 'funasr'

with grpcclient.InferenceServerClient(url='localhost:8001', verbose=False) as client:
    import pdb; pdb.set_trace()
    """
    waveform, _ = librosa.load('../funasr_torch/asr_example.wav', sr=16000)
    input_data = waveform.astype(np.float32).reshape(1, -1)
    """
    from lhotse import CutSet, load_manifest
    filename = '/workspace/aishell-test-dev-manifests/data/fbank/aishell_cuts_test.jsonl.gz'
    cuts = load_manifest(filename)
    cuts_list = cuts.split(60)
    for cuts in cuts_list:
        for i, c in enumerate(cuts):
            input_data = c.load_audio().reshape(1, -1).astype(np.float32)
            break
    # """

    inputs = [
        grpcclient.InferInput(
            'WAV', input_data.shape, np_to_triton_dtype(input_data.dtype)
        )
    ]
    inputs[0].set_data_from_numpy(input_data)
    outputs = [grpcclient.InferRequestedOutput('TEXT')]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    output_data = response.as_numpy('TEXT')
    print(output_data.tolist().decode('utf-8'))
