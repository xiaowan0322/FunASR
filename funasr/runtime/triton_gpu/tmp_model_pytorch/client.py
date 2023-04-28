from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import sys
import numpy as np


model_name = "pytorch"
shape = [4]

with grpcclient.InferenceServerClient(url="localhost:8001", verbose=False) as client:

    input0_data = np.random.rand(*shape).astype(np.float32)
    input1_data = np.random.rand(*shape).astype(np.float32)
    inputs = [
        grpcclient.InferInput(
            "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
        ),
        grpcclient.InferInput(
            "INPUT1", input1_data.shape, np_to_triton_dtype(input1_data.dtype)
        ),
    ]

    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)

    outputs = [
        grpcclient.InferRequestedOutput("OUTPUT0"),
        grpcclient.InferRequestedOutput("OUTPUT1"),
    ]

    response = client.infer(
        model_name, inputs, request_id=str(1), outputs=outputs
    )

    statistics = client.get_inference_statistics(model_name=model_name)
    print(statistics)
    import pdb; pdb.set_trace()

    output0_data = response.as_numpy("OUTPUT0")
    output1_data = response.as_numpy("OUTPUT1")

    print("INPUT0 ({}) + INPUT1 ({}) = OUTPUT0 ({})".format(
        input0_data, input1_data, output0_data)
    )
    print("INPUT0 ({}) - INPUT1 ({}) = OUTPUT0 ({})".format(
        input0_data, input1_data, output1_data)
    )

    if not np.allclose(input0_data + input1_data, output0_data):
        print("pytorch example error: incorrect sum")
        sys.exit(1)

    if not np.allclose(input0_data - input1_data, output1_data):
        print("pytorch example error: incorrect difference")
        sys.exit(1)

    print('PASS: pytorch')
    sys.exit(0)
