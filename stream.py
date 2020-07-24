import argparse
import cv2
import threading

import numpy as np

from host import Host
import di_utils


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

    def run_inference(self, conn, img, frame_num, idx):

        if self.verbose:
            print("Sent frame: ", frame_num)

        self.host.send_image(conn[0], img)
        result = self.host.get_infer_result(conn[0])
        if result is None:
            print("Connection {} broken".format(idx))
            return

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

    def stitch(self):

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

        stitch_thread = threading.Thread(target=self.stitch)
        stitch_thread.start()

        frame_num = 0

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
                    t = threading.Thread(target=self.run_inference, args=(conn, frame, frame_num, idx))
                    t.start()
                    sent = True
                    break

            frame_num += 1

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
