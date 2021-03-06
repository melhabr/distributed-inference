import argparse
import socket
import threading
import struct

import numpy as np
import cv2


class Relay:

    def __init__(self, ip, verbose=False):

        self.verbose = verbose
        s = ip.split(':')
        if len(s) > 1:
            self.ip = s[0]
            self.port = int(s[1])
        else:
            self.ip = s[0]
            self.port = 8080

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.ip, self.port))
        self.images = []
        self.stop = False

        self.buffer_lock = threading.Lock()

        self.t = threading.Thread(target=self._store_images, daemon=True)
        self.t.start()

    def _store_images(self):

        buf = bytearray()
        if self.stop:
            return None

        while not self.stop:

            # Initial phase to get size
            while len(buf) < 4 and not self.stop:
                data = self.socket.recv(4096)
                if not data:
                    print("Source connection closed")
                    self.stop = True
                    return
                buf += data

            remaining = struct.unpack("I", buf[0:4])[0]
            buf = buf[4:]
            remaining -= len(buf)

            print("Receiving image of size ", remaining)

            while remaining > 0:
                data = self.socket.recv(remaining)
                if not data:
                    print("Source connection closed")
                    self.stop = True
                    return
                buf += data
                remaining -= len(data)

            if remaining < 0:
                img = bytes(buf[:remaining])
                buf = buf[remaining:]
            else:
                img = bytes(buf)
                buf = bytearray()

            img = cv2.imdecode(np.fromstring(img, dtype=np.uint8), cv2.IMREAD_COLOR)

            with self.buffer_lock:
                self.images.append(img)

    def get_image(self):

        if self.verbose:
            print("Waiting to grab image")
        while not self.images:
            if self.stop:
                break

        if self.stop:
            return None

        with self.buffer_lock:
            return self.images.pop(0)

    def send_results(self, results):

        self.socket.sendall(struct.pack("B", len(results)))

        for result in results:
            # TODO: optimize by removing padding (is 16, should be 14)
            data = struct.pack("HHHHHf",
                               result[0][0], result[0][1], result[0][2], result[0][3], result[1], result[2])
            self.socket.sendall(data)

    def close(self):
        self.stop = True
        self.t.join()


# Class unit test
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", required=True)
    args = parser.parse_args()

    ir = Relay(args.ip)

    img = ir.get_image()

    if img is not None:
        cv2.imwrite("out.jpg", img)

    ir.close()
