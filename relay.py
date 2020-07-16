import argparse
import socket
import threading
import struct

import numpy as np
import cv2


class Relay:

    def __init__(self, ip, port=45005):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((ip, port))
        self.images = []
        self.stop = False

        self.t = threading.Thread(target=self.store_images, daemon=True)
        self.t.start()

    def store_images(self):

        buf = b''
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
            print("receiving image of size, ", remaining)
            buf = buf[4:]
            remaining -= len(buf)

            while remaining > 0:
                data = self.socket.recv(remaining)
                if not data:
                    print("Source connection closed")
                    self.stop = True
                    return
                buf += data
                remaining -= len(data)

            img = cv2.imdecode(np.fromstring(buf, dtype=np.uint8), 1)
            self.images.append(img)
            buf = b''

    def get_image(self):

        while not self.images:
            if self.stop:
                break

        if self.stop:
            return None

        return self.images.pop(0)

    def send_results(self, results):

        self.socket.sendall(struct.pack("B", len(results)))

        print(results)
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
