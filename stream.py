import argparse
import cv2
import threading
import time

import numpy as np

from host import Host
import di_utils

DEVICE_NAME_MAP = ["EdgeTPU", "Jetson", "UP Squared"]


class StreamMeasurement:
    MODE_SEND, MODE_GET = 0, 1

    def __init__(self, name_map, report_interval=10):

        self.report_interval = report_interval
        self.name_map = name_map

        self.start_times = [0.] * len(name_map)
        self.send_times = [0.] * len(name_map)
        self.get_times = [0.] * len(name_map)
        self.send_recents = [0.] * len(name_map)
        self.get_recents = [0.] * len(name_map)
        self.numread = [0] * len(name_map)

        self.last_report = -1000
        self.start_time = time.time()

    def start(self, device_num):

        if device_num >= len(self.start_times):
            raise ValueError("StreamMeasurement: Requested {} out of {} devices"
                             .format(device_num, len(self.start_times)))

        self.start_times[device_num] = time.time()

    def stop(self, device_num, mode):

        if mode not in range(2):
            raise ValueError("StreamMeasurement: Bad mode")

        if device_num >= len(self.start_times):
            raise ValueError("StreamMeasurement: Requested {} out of {} devices"
                             .format(device_num, len(self.start_times)))

        if mode == StreamMeasurement.MODE_SEND:

            self.send_recents[device_num] = time.time() - self.start_times[device_num]

            self.send_times[device_num] = \
                self.send_times[device_num] + (self.send_recents[device_num] - self.send_times[device_num]) / (
                        self.numread[device_num] + 1)

        elif mode == StreamMeasurement.MODE_GET:

            self.get_recents[device_num] = time.time() - self.start_times[device_num]

            self.get_times[device_num] = \
                self.get_times[device_num] + (self.get_recents[device_num] - self.get_times[device_num]) / (
                        self.numread[device_num] + 1)

            self.numread[device_num] += 1

    def report(self, force=False):

        if sum(self.numread) == 0:
            return False

        if time.time() - self.last_report > self.report_interval or force:
            print("----------------------------")
            print("Average transmission statistics:")
            for i, name in enumerate(self.name_map):
                print(
                    "{}: Send: {}ms (recent {}ms) | Get: {}ms (recent {}ms) | Total: {}ms (recent {}ms)"
                    .format(name, round(self.send_times[i] * 1000, 3), round(self.send_recents[i] * 1000, 3),
                            round(self.get_times[i] * 1000, 3), round(self.get_recents[i] * 1000, 3),
                            round((self.get_times[i] + self.send_times[i]) * 1000, 3),
                            round((self.get_recents[i] + self.send_recents[i]) * 1000, 3)))
            print("Current FPS:", sum(self.numread)/ (time.time() - self.start_time))
            print("Distribution:",
                  " ".join(["{}: {}%".format(self.name_map[i], round(self.numread[i] * 100 / sum(self.numread), 2))
                            for i in range(len(self.name_map))]))
            self.last_report = time.time()
            return True

        return False


class DistributedStream:

    def __init__(self, port, labels, verbose=False):

        self.labels = None
        if labels:
            self.labels = di_utils.read_labels(labels)
        self.host = Host(port, verbose)
        self.ready = [True] * len(self.host.conns)
        self.labeled_frames = []
        self.buffer_lock = threading.Lock()
        self.final_frame = -1
        self.w = -1
        self.h = -1
        self.verbose = verbose
        self.watch = StreamMeasurement(DEVICE_NAME_MAP)

    def _run_inference(self, conn, img, frame_num, idx):

        self.watch.start(idx)
        self.host.send_image(conn[0], img)
        if self.verbose:
            print("Sent frame: ", frame_num)
        self.watch.stop(idx, StreamMeasurement.MODE_SEND)

        self.watch.start(idx)
        result = self.host.get_infer_result(conn[0])
        if result is None:
            print("Connection {} broken".format(idx))
            return
        self.watch.stop(idx, StreamMeasurement.MODE_GET)

        if self.verbose:
            print("Got infer result for frame ", frame_num)

        labeled_img = np.copy(img)

        result = [((int(x1 * self.w / 300),
                    int(y1 * self.h / 300),
                    int(x2 * self.w / 300),
                    int(y2 * self.h / 300), cls, conf)) for (x1, y1, x2, y2, cls, conf) in result]

        if self.labels:
            di_utils.draw_labels(labeled_img, result, labels=self.labels)
        else:
            di_utils.draw_labels(labeled_img, result)

        with self.buffer_lock:
            self.labeled_frames.append((frame_num, labeled_img))
        self.ready[idx] = True

    def _stitch(self):

        done = False
        current_frame = 0
        vw = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (self.w, self.h))

        while not done:

            for frame in self.labeled_frames:
                if frame[0] == current_frame:
                    vw.write(frame[1])
                    with self.buffer_lock:
                        del frame
                    current_frame += 1
                    if self.verbose:
                        print("Stitched frame ", current_frame - 1)

            if self.final_frame != -1:
                if current_frame == self.final_frame:
                    done = True

    def stream_video(self, input):

        vcap = cv2.VideoCapture(input)
        self.w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        stitch_thread = threading.Thread(target=self._stitch)
        stitch_thread.start()

        frame_num = 0
        self.watch.start_time = time.time()

        while vcap.isOpened():

            ret, img = vcap.read()

            if not ret:
                break

            if (self.w, self.h) != (300, 300):
                frame = cv2.resize(img, (300, 300))
            else:
                frame = img

            sent = False

            while not sent:

                for idx, conn in enumerate(self.host.get_connections()):

                    if not self.ready[idx]:
                        continue

                    self.ready[idx] = False
                    t = threading.Thread(target=self._run_inference, args=(conn, frame, frame_num, idx))
                    t.start()
                    sent = True
                    break

            frame_num += 1

            self.watch.report()

        self.watch.report(force=True)

        self.final_frame = frame_num - 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="File path to input image", required=True)
    parser.add_argument("--port", "-p", help="Listening port to use", default=8080, type=int)
    parser.add_argument("--labels", "-l", help="Path of labels file")
    parser.add_argument("--verbose", "-v", help="Enables verbose logging", action='store_true')
    args = parser.parse_args()

    stream = DistributedStream(args.port, args.labels, args.verbose)
    stream.stream_video(args.input)


if __name__ == '__main__':
    main()
