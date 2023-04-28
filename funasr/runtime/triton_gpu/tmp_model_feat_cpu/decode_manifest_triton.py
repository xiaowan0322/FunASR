import argparse
import asyncio
import math
import time
import types
from pathlib import Path
import json
import numpy as np
import tritonclient
import tritonclient.grpc.aio as grpcclient
from lhotse import CutSet, load_manifest
from tritonclient.utils import np_to_triton_dtype

# from icefall.utils import store_transcripts, write_error_stats

DEFAULT_MANIFEST_FILENAME = "/mnt/samsung-t7/yuekai/aishell-test-dev-manifests/data/fbank/aishell_cuts_test.jsonl.gz"  # noqa


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--server-addr",
        type=str,
        default="localhost",
        help="Address of the server",
    )

    parser.add_argument(
        "--server-port",
        type=int,
        default=8001,
        help="Port of the server",
    )

    parser.add_argument(
        "--manifest-filename",
        type=str,
        default=DEFAULT_MANIFEST_FILENAME,
        help="Path to the manifest for decoding",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="transducer",
        help="triton model_repo module name to request",
    )

    parser.add_argument(
        "--num-tasks",
        type=int,
        default=50,
        help="Number of tasks to use for sending",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=5,
        help="Controls how frequently we print the log.",
    )

    parser.add_argument(
        "--compute-cer",
        action="store_true",
        default=False,
        help="""True to compute CER, e.g., for Chinese.
        False to compute WER, e.g., for English words.
        """,
    )

    parser.add_argument(
        "--stats-file",
        type=str,
        required=False,
        default="./stats.json",
        help="output of stats anaylasis",
    )

    return parser.parse_args()


async def send(
    cuts: CutSet,
    name: str,
    triton_client: tritonclient.grpc.aio.InferenceServerClient,
    protocol_client: types.ModuleType,
    log_interval: int,
    compute_cer: bool,
    model_name: str,
):
    total_duration = 0.0
    results = []

    for i, c in enumerate(cuts):
        if i % log_interval == 0:
            print(f"{name}: {i}/{len(cuts)}")

        waveform = c.load_audio().reshape(-1).astype(np.float32)
        sample_rate = 16000

        # padding to nearset 10 seconds
        samples = np.zeros(
            (
                1,
                10 * sample_rate * (int(len(waveform) / sample_rate // 10) + 1),
            ),
            dtype=np.float32,
        )
        samples[0, : len(waveform)] = waveform

        lengths = np.array([[len(waveform)]], dtype=np.int32)

        inputs = [
            protocol_client.InferInput(
                "WAV", samples.shape, np_to_triton_dtype(samples.dtype)
            ),
        ]
        inputs[0].set_data_from_numpy(samples)
        outputs = [protocol_client.InferRequestedOutput("TEXT")]
        sequence_id = 10086 + i

        response = await triton_client.infer(
            model_name, inputs, request_id=str(sequence_id), outputs=outputs
        )

        # decoding_results = response.as_numpy("TEXT")[0]
        decoding_results = response.as_numpy("TEXT")
        if type(decoding_results) == np.ndarray:
            # decoding_results = b" ".join(decoding_results).decode("utf-8")
            decoding_results = decoding_results.tolist().decode("utf-8")
        else:
            # For wenet
            decoding_results = decoding_results.decode("utf-8")

        total_duration += c.duration

        # print(decoding_results)
        if compute_cer:
            ref = c.supervisions[0].text.split()
            hyp = decoding_results.split()
            ref = list("".join(ref))
            hyp = list("".join(hyp))
            results.append((c.id, ref, hyp))
        else:
            results.append(
                (
                    c.id,
                    c.supervisions[0].text.split(),
                    decoding_results.split(),
                )
            )  # noqa

    return total_duration, results


async def main():
    args = get_args()
    filename = args.manifest_filename
    server_addr = args.server_addr
    server_port = args.server_port
    url = f"{server_addr}:{server_port}"
    num_tasks = args.num_tasks
    log_interval = args.log_interval
    compute_cer = args.compute_cer

    cuts = load_manifest(filename)
    cuts_list = cuts.split(num_tasks)
    # cuts_list = cuts_list * 2
    # import pdb; pdb.set_trace()
    tasks = []

    triton_client = grpcclient.InferenceServerClient(url=url, verbose=False)
    protocol_client = grpcclient

    start_time = time.time()
    for i in range(num_tasks):
        task = asyncio.create_task(
            send(
                cuts=cuts_list[i],
                name=f"task-{i}",
                triton_client=triton_client,
                protocol_client=protocol_client,
                log_interval=log_interval,
                compute_cer=compute_cer,
                model_name=args.model_name,
            )
        )
        tasks.append(task)

    ans_list = await asyncio.gather(*tasks)

    end_time = time.time()
    elapsed = end_time - start_time

    results = []
    total_duration = 0.0
    latency_data = []
    for ans in ans_list:
        total_duration += ans[0]
        results += ans[1]

    rtf = elapsed / total_duration

    print("Speech: {:.4f}".format(1.0 / rtf))
    s = f"RTF: {rtf:.4f}\n"
    s += f"total_duration: {total_duration:.3f} seconds\n"
    s += f"({total_duration/3600:.2f} hours)\n"
    s += (
        f"processing time: {elapsed:.3f} seconds "
        f"({elapsed/3600:.2f} hours)\n"
    )

    print(s)

    with open("rtf.txt", "w") as f:
        f.write(s)

    """
    name = Path(filename).stem.split(".")[0]
    results = sorted(results)
    store_transcripts(filename=f"recogs-{name}.txt", texts=results)

    with open(f"errs-{name}.txt", "w") as f:
        write_error_stats(f, "test-set", results, enable_log=True)

    with open(f"errs-{name}.txt", "r") as f:
        print(f.readline())  # WER
        print(f.readline())  # Detailed errors

    if args.stats_file:
        stats = await triton_client.get_inference_statistics(
            model_name="", as_json=True
        )
        with open(args.stats_file, "w") as f:
            json.dump(stats, f)
    """


if __name__ == "__main__":
    asyncio.run(main())
