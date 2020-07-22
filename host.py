import socket
import threading
import argparse
import cv2
import struct
import numpy as np
import time

import di_utils

class Host:

    def __init__(self, port=8080):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind(('', port))
        self.s.listen()
        self.close_new = False
        self.conns = []

        t = threading.Thread(target=self.add_connections, daemon=True)
        t.start()
        self.first_rec_time = 0

        input("Press enter to stop accepting connections and begin transmission\n")
        self.close_new = True

    def add_connections(self):
        while True:
            res = self.s.accept()
            if not self.close_new:
                self.conns.append(res)
                print("Added a new connection, {}. {} connections total"
                      .format(res, len(self.conns)))

    def remove_connection(self, conn):
        self.conns.remove(conn)

    def close(self):
        self.close_new = True
        for conn in self.conns:
            conn[0].close()
        self.s.close()

    def get_infer_result(self, conn):
        buf = b''
        results = []

        data = conn.recv(4096)
        self.first_rec_time = time.time()
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

    def send_image(self, conn, img):
        data = cv2.imencode(".png", img)[1].tostring()
        conn.sendall(struct.pack("I", len(data)) + data)

    def get_connections(self):
        return Host.ConnectionIterator(self.conns.copy())

    class ConnectionIterator:
        def __init__(self, conns):
            self.contents = conns
            pass

        def __iter__(self):
            self.n = 0
            return self

        def __next__(self):
            if self.n >= len(self.contents):
                raise StopIteration
            obj = self.contents[self.n]
            self.n += 1
            if not obj:
                return self.__next__()
            return obj




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="File path to input image", required=True)
    parser.add_argument("--port", "-p", help="Listening port to use", default=8080, type=int)
    parser.add_argument("--labels", "-l", help="Path of labels file")
    args = parser.parse_args()

    host = Host(args.port)

    if args.labels:
        labels = di_utils.read_labels(args.labels)

    img = cv2.imread(args.input)
    ih, iw = img.shape[:-1]
    if (iw, ih) != (300, 300):
        frame = cv2.resize(img, (300, 300))
    else:
        frame = img

    for idx, conn in enumerate(host.get_connections()):
        send_start = time.time()
        host.send_image(conn[0], frame)
        send_end = time.time()
        result = host.get_infer_result(conn[0])
        get_end = time.time()
        if result is None:
            host.remove_connection(conn)
            continue
        print("Connection {}: Send: {}ms | Gap: {}ms | Get: {}ms | Total: {}ms"
              .format(idx, round((send_end - send_start) * 1000, 3), round((host.first_rec_time - send_end) * 1000, 3),
                      round((get_end - host.first_rec_time) * 1000, 3), round((get_end - send_start) * 1000, 3)))

        labeled_img = np.copy(img)

        result = [((int(x1 * iw / 300),
                    int(y1 * ih / 300),
                    int(x2 * iw / 300),
                    int(y2 * ih / 300), cls, conf)) for (x1, y1, x2, y2, cls, conf) in result]

        if args.labels:
            di_utils.draw_labels(labeled_img, result, labels=labels)
        else:
            di_utils.draw_labels(labeled_img, result)
        cv2.imwrite("out/out{}.jpg".format(idx), labeled_img)

    while True:
        pass  # The program needs to stay alive to allow for TCP ping measurement

    sp.close()


if __name__ == '__main__':
    main()
