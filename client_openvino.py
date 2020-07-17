#!/usr/bin/env python
"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

# This script is a substantially modified version of the one here:
# https://github.com/openvinotoolkit/openvino/tree/master/inference-engine/ie_bridges/python/sample/object_detection_sample_ssd

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore, IENetwork

import time
from tcp_latency import measure_latency

from relay import Relay

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group("Options")
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("--ip", help="Required. Path to video file.",
                      required=True, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "Absolute path to a shared library with the kernels implementations.",
                      type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; "
                           "CPU, GPU, FPGA or MYRIAD is acceptable. "
                           "Sample will look for a suitable plugin for device specified (CPU by default)",
                      default="CPU", type=str)
    args.add_argument("-nt", "--number_top", help="Optional. Number of top results", default=10, type=int)
    args.add_argument('--report_interval', '-r', help="Duration of reporting interval, in seconds", default=10,
                      type=int)
    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    log.info("Loading Inference Engine")
    ie = IECore()
    # --------------------------- 1. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)
    # -----------------------------------------------------------------------------------------------------

    # ------------- 2. Load Plugin for inference engine and extensions library if specified --------------
    log.info("Device info:")
    versions = ie.get_versions(args.device)
    print("{}{}".format(" " * 8, args.device))
    print("{}MKLDNNPlugin version ......... {}.{}".format(" " * 8, versions[args.device].major,
                                                          versions[args.device].minor))
    print("{}Build ........... {}".format(" " * 8, versions[args.device].build_number))

    if args.cpu_extension and "CPU" in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
        log.info("CPU extension loaded: {}".format(args.cpu_extension))

    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    # -----------------------------------------------------------------------------------------------------

    # -----------------------------------------3. Prepare input blobs ------------------------------------

    log.info("Preparing input blobs")
    print("inputs number: " + str(len(net.inputs.keys())))

    for input_key in net.inputs:
        print("input shape: " + str(net.inputs[input_key].shape))
        print("input key: " + input_key)
        if len(net.inputs[input_key].layout) == 4:
            n, c, h, w = net.inputs[input_key].shape

    assert (len(net.inputs.keys()) == 1 or len(
        net.inputs.keys()) == 2), "Sample supports topologies only with 1 or 2 inputs"
    out_blob = next(iter(net.outputs))
    input_name, input_info_name = "", ""

    for input_key in net.inputs:
        if len(net.inputs[input_key].layout) == 4:
            input_name = input_key
            log.info("Batch size is {}".format(net.batch_size))
            net.inputs[input_key].precision = 'U8'
        elif len(net.inputs[input_key].layout) == 2:
            input_info_name = input_key
            net.inputs[input_key].precision = 'FP32'
            if net.inputs[input_key].shape[1] != 3 and net.inputs[input_key].shape[1] != 6 or \
                    net.inputs[input_key].shape[0] != 1:
                log.error('Invalid input info. Should be 3 or 6 values length.')

    data = {}

    if input_info_name != "":
        infos = np.ndarray(shape=(n, c), dtype=float)
        for i in range(n):
            infos[i, 0] = h
            infos[i, 1] = w
            infos[i, 2] = 1.0
        data[input_info_name] = infos

    # ---------------------------------------- 4. Prepare output blobs ------------------------------------
    log.info('Preparing output blobs')

    output_name, output_info = "", net.outputs[next(iter(net.outputs.keys()))]
    for output_key in net.outputs:
        if net.layers[output_key].type == "DetectionOutput":
            output_name, output_info = output_key, net.outputs[output_key]

    if output_name == "":
        log.error("Can't find a DetectionOutput layer in the topology")

    output_dims = output_info.shape
    if len(output_dims) != 4:
        log.error("Incorrect output dimensions for SSD model")
    max_proposal_count, object_size = output_dims[2], output_dims[3]

    if object_size != 7:
        log.error("Output item should have 7 as a last dimension")

    output_info.precision = "FP32"

    # --------------------------- 5. Read, preproccess, inference, and write simultaneously ----------------------

    log.info("Loading model to the device")
    exec_net = ie.load_network(network=net, device_name=args.device)
    log.info("Running inference")

    relay = Relay(args.ip)

    numread = 0
    avg_preproc_time_c = 0.
    avg_preproc_time_w = 0.
    avg_infer_time_c = 0.
    avg_infer_time_w = 0.
    avg_postproc_time_c = 0.
    avg_postproc_time_w = 0.
    display_timer = -1000
    start_time = time.time()

    while True:

        img = relay.get_image()

        if img is None:
            break

        start_w = time.time()
        start_c = time.process_time()
        ih, iw = img.shape[:-1]
        if (ih, iw) != (h, w):
            img = cv2.resize(img, (w, h))
        img = img.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        end_c = time.process_time()
        end_w = time.time()

        avg_preproc_time_w = avg_preproc_time_w + ((end_w - start_w) - avg_preproc_time_w) / (numread + 1)
        avg_preproc_time_c = avg_preproc_time_c + ((end_c - start_c) - avg_preproc_time_c) / (numread + 1)

        data[input_name] = img
        start_w = time.time()
        start_c = time.process_time()
        res = exec_net.infer(inputs=data)
        res = res[out_blob]
        res = res[0][0]
        end_c = time.process_time()
        end_w = time.time()

        avg_infer_time_w = avg_infer_time_w + ((end_w - start_w) - avg_infer_time_w) / (numread + 1)
        avg_infer_time_c = avg_infer_time_c + ((end_c - start_c) - avg_infer_time_c) / (numread + 1)
        recent_infer_time_w = end_w - start_w

        start_w = time.time()
        start_c = time.process_time()

        results = []
        for proposal in res:

            if int(proposal[1]) == 0:
                continue

            box = [0] * 4

            box[0] = np.int(iw * proposal[3])
            box[1] = np.int(ih * proposal[4])
            box[2] = np.int(iw * proposal[5])
            box[3] = np.int(ih * proposal[6])

            result = (box, int(proposal[1]), proposal[2])
            results.append(result)

        relay.send_results(results)

        end_c = time.process_time()
        end_w = time.time()

        avg_postproc_time_w = avg_postproc_time_w + ((end_w - start_w) - avg_postproc_time_w) / (numread + 1)
        avg_postproc_time_c = avg_postproc_time_c + ((end_c - start_c) - avg_postproc_time_c) / (numread + 1)

        if time.time() - display_timer > args.report_interval:
            display_timer = time.time()
            print("--------------------------------")
            print("Average time of preprocessing - CPU: {}ms, wall: {}ms".format(
                round(avg_preproc_time_c * 1000, 3), round(avg_preproc_time_w * 1000, 3)))
            print("Average time of inference - CPU: {}ms, wall: {}ms (recent: {}ms)".format(
                round(avg_infer_time_c * 1000, 3), round(avg_infer_time_w * 1000, 3),
                round(recent_infer_time_w * 1000, 3)))
            print("Average time of postprocessing - CPU: {}ms, wall: {}ms".format(
                round(avg_postproc_time_c * 1000, 3), round(avg_postproc_time_w * 1000, 3)))
            print("Current FPS: ", numread / (time.time() - start_time))
            print("TCP Latency to source: ", round(measure_latency(host=args.ip, port=relay.port)[0], 3), "ms")

        numread += 1

    relay.close()

    # -----------------------------------------------------------------------------------------------------

    log.info("Execution successful\n")


if __name__ == '__main__':
    sys.exit(main() or 0)
