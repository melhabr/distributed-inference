# Inference code significantly adapted from JK Jung. MIT Licence
# See https://github.com/jkjung-avt/tensorrt_demos

import argparse
import cv2
import time
import numpy as np

import pycuda.autoinit
from tcp_latency import measure_latency
import pycuda.driver as cuda
import tensorrt as trt

from relay import Relay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--ip", type=str)
    parser.add_argument("--verbose", "-v", action='store_true')
    parser.add_argument('--report_interval', '-r', help="Duration of reporting interval, in seconds", default=10,
                        type=int)
    args = parser.parse_args()

    relay = Relay(args.ip)

    # Initialize TRT environment
    input_shape = (300, 300)
    trt_logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(trt_logger, '')
    with open(args.model, 'rb') as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    host_inputs = []
    cuda_inputs = []
    host_outputs = []
    cuda_outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        host_mem = cuda.pagelocked_empty(size, np.float32)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(cuda_mem))
        if engine.binding_is_input(binding):
            host_inputs.append(host_mem)
            cuda_inputs.append(cuda_mem)
        else:
            host_outputs.append(host_mem)
            cuda_outputs.append(cuda_mem)
    context = engine.create_execution_context()


    numread = 0
    avg_preproc_time_c = 0.
    avg_preproc_time_w = 0.
    avg_infer_time_c = 0.
    avg_infer_time_w = 0.
    avg_postproc_time_c = 0.
    avg_postproc_time_w = 0.
    display_timer = -1000

    while True:

        img = relay.get_image()

        if img is None:
            break

        # Preprocessing:
        start_w = time.time()
        start_c = time.process_time()

        ih, iw = img.shape[:-1]
        if (iw, ih) != input_shape:
            img = cv2.resize(img, input_shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img *= (2.0 / 255.0)
        img -= 1.0

        np.copyto(host_inputs[0], img.ravel())
        end_c = time.process_time()
        end_w = time.time()

        avg_preproc_time_c = avg_preproc_time_c + ((end_c - start_c) - avg_preproc_time_c) / (numread + 1)
        avg_preproc_time_w = avg_preproc_time_w + ((end_w - start_w) - avg_preproc_time_w) / (numread + 1)

        start_w = time.time()
        start_c = time.process_time()
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[1], cuda_outputs[1], stream)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()
        end_c = time.process_time()
        end_w = time.time()

        avg_infer_time_c = avg_infer_time_c + ((end_c - start_c) - avg_infer_time_c) / (numread + 1)
        avg_infer_time_w = avg_infer_time_w + ((end_w - start_w) - avg_infer_time_w) / (numread + 1)

        # Postprocessing
        start_w = time.time()
        start_c = time.process_time()
        output = host_outputs[0]
        results = []
        for prefix in range(0, len(output), 7):

            conf = float(output[prefix + 2])
            if conf < 0.5:
                continue
            x1 = int(output[prefix + 3] * iw)
            y1 = int(output[prefix + 4] * ih)
            x2 = int(output[prefix + 5] * iw)
            y2 = int(output[prefix + 6] * ih)
            cls = int(output[prefix + 1])
            results.append(((x1, y1, x2, y2), cls, conf))

        if args.verbose:
            print(results)

        relay.send_results(results)

        end_c = time.process_time()
        end_w = time.time()

        avg_postproc_time_c = avg_postproc_time_c + ((end_c - start_c) - avg_postproc_time_c) / (numread + 1)
        avg_postproc_time_w = avg_postproc_time_w + ((end_w - start_w) - avg_postproc_time_w) / (numread + 1)

        numread += 1

        if time.time() - display_timer > args.report_interval:
            display_timer = time.time()
            print("--------------------------------")
            print("Average time of preprocessing - CPU: {}ms, wall: {}ms".format(
                round(avg_preproc_time_c * 1000, 3), round(avg_preproc_time_w * 1000, 3)))
            print("Average time of inference - CPU: {}ms, wall: {}ms".format(
                round(avg_infer_time_c * 1000, 3), round(avg_infer_time_w * 1000, 3)))
            print("Average time of postprocessing - CPU: {}ms, wall: {}ms".format(
                round(avg_postproc_time_c * 1000, 3), round(avg_postproc_time_w * 1000, 3)))
            print("TCP Latency to source: ", round(measure_latency(host=args.ip, port=relay.port)[0], 3), "ms")

        relay.close()




if __name__ == '__main__':
    main()
