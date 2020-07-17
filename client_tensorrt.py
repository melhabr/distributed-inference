# Inference code significantly adapted from JK Jung. MIT Licence
# See https://github.com/jkjung-avt/tensorrt_demos

import argparse
import cv2
import time

import pycuda.autoinit
from tcp_latency import measure_latency

from utils.ssd_mod import TrtSSD

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

    trt_ssd = TrtSSD(args.model, (300, 300))

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

        (boxes, confs, classes), times = trt_ssd.detect(img, 0.5)

        avg_preproc_time_c = avg_preproc_time_c + (times[0] - avg_preproc_time_c) / (numread + 1)
        avg_preproc_time_w = avg_preproc_time_w + (times[1] - avg_preproc_time_w) / (numread + 1)
        avg_infer_time_c = avg_infer_time_c + (times[2] - avg_infer_time_c) / (numread + 1)
        avg_infer_time_w = avg_infer_time_w + (times[3] - avg_infer_time_w) / (numread + 1)

        start_w = time.time()
        start_c = time.process_time()
        if args.verbose:
            print('identified class:', classes[0])
            print('box: ', boxes[0])

        results = [(box, id, conf) for box, conf, id in zip(boxes, confs, classes)]
        relay.send_results(results)

        end_c = time.process_time()
        end_w = time.time()

        avg_postproc_time_c = avg_postproc_time_c + ((end_c - start_c + times[4]) - avg_postproc_time_c) / (numread + 1)
        avg_postproc_time_w = avg_postproc_time_w + ((end_w - start_w + times[5]) - avg_postproc_time_w) / (numread + 1)

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
            print("Current FPS: ", numread / (time.time() - start_time))
            print("TCP Latency to source: ", round(measure_latency(host=args.ip, port=relay.port)[0], 3), "ms")

        relay.close()


if __name__ == '__main__':
    main()
