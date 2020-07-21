# Inference code in file was adapted from https://gist.github.com/aallan/6778b5a046aef3a76966e335ca186b7f

import argparse
import cv2

from tcp_latency import measure_latency

from edgetpu.detection.engine import DetectionEngine
from relay import Relay
from stopwatch import Stopwatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path of the detection model.', required=True, type=str)
    parser.add_argument('--ip', "-i", help='File path of the input image.', required=True, type=str)
    parser.add_argument('--report_interval', '-r', help="Duration of reporting interval, in seconds", default=10,
                        type=int)
    parser.add_argument('-v', "--verbose", help="Print information about detected objects", action='store_true')
    args = parser.parse_args()

    relay = Relay(args.ip)

    engine = DetectionEngine(args.model)

    watch = Stopwatch()

    while True:

        img = relay.get_image()

        if img is None:
            break

        watch.start()
        initial_h, initial_w, _ = img.shape
        if (initial_h, initial_w) != (300, 300):
            frame = cv2.resize(img, (300, 300))
        else:
            frame = img
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        watch.stop(Stopwatch.MODE_PREPROCESS)

        watch.start()
        ans = engine.detect_with_input_tensor(frame.flatten(), threshold=0.5, top_k=10)
        watch.stop(Stopwatch.MODE_INFER)

        watch.start()
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

        watch.stop(Stopwatch.MODE_POSTPROCESS)

        if watch.report():
            print("TCP Latency to source: ", round(measure_latency(host=args.ip, port=relay.port)[0], 3), "ms")

    relay.close()


if __name__ == '__main__':
    main()
