# Inference code in file was adapted from https://gist.github.com/aallan/6778b5a046aef3a76966e335ca186b7f

import argparse
import cv2
import os
import numpy as np
import time

from PIL import Image
from PIL import ImageFont, ImageDraw, ImageColor
from tcp_latency import measure_latency

from edgetpu.detection.engine import DetectionEngine
from relay import Relay
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path of the detection model.', required=True, type=str)
    parser.add_argument('--ip', "-i", help='File path of the input image.', required=True, type=str)
    parser.add_argument('--report_interval', '-r', help="Duration of reporting interval, in seconds", default=10, type=int)
    parser.add_argument('-v', "--verbose", help="Print information about detected objects", action='store_true')
    args = parser.parse_args()

    relay = Relay(args.ip)

    engine = DetectionEngine(args.model)

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

        start_w = time.time()
        start_c = time.process_time()
        initial_h, initial_w, _ = img.shape
        if (initial_h, initial_w) != (300, 300):
            frame = cv2.resize(img, (300, 300))
        else:
            frame = img
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        end_c = time.process_time()
        end_w = time.time()

        avg_preproc_time_w = avg_preproc_time_w + ((end_w - start_w) - avg_preproc_time_w)/(numread + 1)
        avg_preproc_time_c = avg_preproc_time_c + ((end_c - start_c) - avg_preproc_time_c)/(numread + 1)
        
        start_w = time.time()
        start_c = time.process_time()
        ans = engine.detect_with_input_tensor(frame.flatten(), threshold=0.5, top_k=10)
        end_c = time.process_time()
        end_w = time.time()

        avg_infer_time_w = avg_infer_time_w + ( (end_w - start_w) - avg_infer_time_w)/(numread + 1)
        avg_infer_time_c = avg_infer_time_c + ( (end_c - start_c) - avg_infer_time_c)/(numread + 1)

        start_w = time.time()
        start_c = time.process_time()    
        # Display result
        if ans:
            results = []
            for obj in ans:

                box = obj.bounding_box.flatten().tolist()
                bbox = [0] * 4
                bbox[0] = int(box[0] * initial_w)
                bbox[1] = int(box[1] * initial_h)
                bbox[2] = int(box[2] * initial_w)
                bbox[3] = int(box[3] * initial_h)

                result = (bbox, obj.label_id + 1, obj.score)
                results.append(result)

            relay.send_results(results)

        end_c = time.process_time()
        end_w = time.time()

        avg_postproc_time_w = avg_postproc_time_w + ( (end_w - start_w) - avg_postproc_time_w)/(numread + 1)
        avg_postproc_time_c = avg_postproc_time_c + ( (end_c - start_c) - avg_postproc_time_c)/(numread + 1)    

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
