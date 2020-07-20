import socket
import threading
import argparse
import cv2
import struct
import numpy as np
import time

import di_utils

args = None


class SocketPool:

    def __init__(self, port):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind(('', port))
        self.s.listen()
        self.close_new = False
        self.conns = []

        t = threading.Thread(target=self.add_connections, daemon=True)
        t.start()

    def add_connections(self):
        while True:
            res = self.s.accept()
            if not self.close_new:
                self.conns.append(res)
                print("Added a new connection, {}. {} connections total"
                      .format(res, len(self.conns)))

    def get_connections(self):
        return self.conns.copy()

    def remove_connection(self, conn):
        self.conns.remove(conn)

    def close(self):
        self.close_new = True
        for conn in self.conns:
            conn[0].close()
        self.s.close()


def get_infer_result(conn):
    buf = b''
    results = []

    data = conn.recv(4096)
    if data == b'':
        return None
    num_results = data[0]

    if len(data) == 1:
        buf = b''
    else:
        data = data[1:]
        buf += data

    while num_results > 0:

        data = conn.recv(4096)
        if data == b'':
            print("ERROR: Inference signal stopped mid-transmission")
            return

        buf += data
        while len(buf) >= 16:
            result = struct.unpack("HHHHHf", buf[0:16])
            results.append(result)
            if len(buf) == 16:
                buf = b''
            else:
                buf = buf[16:]
            num_results -= 1

    return results


def send_image(conn, img):
    data = cv2.imencode(".png", img)[1].tostring()
    conn.sendall(struct.pack("I", len(data)) + data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="File path to input image", required=True)
    parser.add_argument("--port", "-p", help="Listening port to use", default=8080, type=int)
    parser.add_argument("--labels", "-l", help="Path of labels file")
    args = parser.parse_args()

    if args.labels:
        labels = di_utils.read_labels(args.labels)
    sp = SocketPool(args.port)

    input("Press enter to stop accepting connections and begin transmission\n")
    sp.close_new = True

    img = cv2.imread(args.input)
    ih, iw = img.shape[:-1]
    if (iw, ih) != (300, 300):
    #if False:
        frame = cv2.resize(img, (300, 300))
    else:
        frame = img

    for idx, conn in enumerate(sp.get_connections()):
        start = time.time()
        send_image(conn[0], frame)
        result = get_infer_result(conn[0])
        if result is None:
            sp.remove_connection(conn)
            continue
        end = time.time()
        print("Connection {} took time {}ms".format(idx, round((end - start) * 1000, 3)))
        labeled_img = np.copy(img)
        for prop in result:
            adj = (int(prop[0] * iw / 300),
                   int(prop[1] * ih / 300),
                   int(prop[2] * iw / 300),
                   int(prop[3] * ih / 300),)
            di_utils.draw_label(labeled_img, (adj[0], adj[1], adj[2], adj[3]),
                                prop[4], prop[5], labels=labels)
        cv2.imwrite("out/out{}.jpg".format(idx), labeled_img)

    while True:
        pass # The program needs to stay alive to allow for TCP ping measurement

    sp.close()


if __name__ == '__main__':
    main()
