import socket
import threading
import argparse
import cv2
import struct
import numpy as np

import di_utils

args = None

class SocketPool:

    def __init__(self, port):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind(('', port))
        self.s.listen()
        self.stop = False
        self.conns = []

        t = threading.Thread(target=self.add_connections, daemon=True)
        t.start()

    def add_connections(self):
        while not self.stop:
            res = self.s.accept()
            self.conns.append(res)
            print("Added a new connection, {}. {} connections total"
                  .format(res, len(self.conns)))

    def close(self):
        stop = True
        for conn in self.conns:
            conn[0].close()
        self.s.close()

def get_infer_result(conn):

    buf = b''
    results = []

    data = conn.recv(4096)
    if data is None:
        return None

    num_results = data[0]

    if len(data) == 1:
        buf = b''
    else:
        data = data[1:]
        buf += data

    while num_results > 0:

        data = conn.recv(4096)
        if data is None:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="File path to input image", required=True)
    parser.add_argument("--port", "-p", help="Listening port to use", default=45005, type=int)
    parser.add_argument("--labels", "-l", help="Path of labels file")
    args = parser.parse_args()

    if args.labels:
        labels = di_utils.read_labels(args.labels)
    sp = SocketPool(args.port)

    input("Press enter to stop accepting connections and begin transmission\n")
    sp.stop = True

    img = cv2.imread(args.input)
    data = cv2.imencode(".jpg", img)[1].tostring()

    results = []
    for conn in sp.conns:
        print("sending image of size ", len(data))
        conn[0].sendall(struct.pack("I", len(data)) + data)
        result = get_infer_result(conn[0])
        result = [prop for prop in result if prop[4] != 0] # strip out background results
        results.append(result)

    for idx, result in enumerate(results):
        labeled_img = np.copy(img)
        for prop in result:
            di_utils.draw_label(labeled_img, (prop[0], prop[1], prop[2], prop[3]),
                                prop[4], prop[5], labels=labels)
        cv2.imwrite("out/out{}.jpg".format(idx), labeled_img)

    sp.close()


if __name__ == '__main__':
    main()
