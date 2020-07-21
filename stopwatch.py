import time

class Stopwatch:

    MODE_PREPROCESS = 0
    MODE_INFER = 1
    MODE_POSTPROCESS = 2

    def __init__(self, report_interval=10):
        self.report_interval = report_interval
        self.numread = 0
        self.wall_times = [0., 0., 0.]
        self.clock_times = [0., 0., 0.]
        self.recents_w = [0., 0., 0.]
        self.recents_c = [0., 0., 0.]
        self.last_report = -1000

        self.start_c = 0.
        self.start_w = 0.

    def start(self):
        self.start_w = time.time()
        self.start_c = time.process_time()

    def stop(self, mode):

        if mode not in [0, 1, 2]:
            raise ValueError("Stopwatch: Bad watch mode")

        self.recents_c[mode] = time.process_time() - self.start_c
        self.recents_w[mode] = time.time() - self.start_w

        self.wall_times[mode] = self.wall_times[mode] + (self.recents_w[mode] - self.wall_times[mode]) / (self.numread + 1)
        self.clock_times[mode] = self.clock_times[mode] + (self.recents_c[mode] - self.clock_times[mode]) / (self.numread + 1)

        if mode == Stopwatch.MODE_POSTPROCESS:
            self.numread += 1

    def report(self):

        if time.time() - self.last_report > self.report_interval:
            self.last_report = time.time()
            print("--------------------------------")
            print("Average time of preprocessing - CPU: {}ms (recent: {}ms), wall: {}ms (recent: {}ms)".format(
                round(self.clock_times[0] * 1000, 3), round(self.recents_c[0] * 1000, 3), round(self.wall_times[0] * 1000, 3), round(self.recents_w[0] * 1000, 3)))
            print("Average time of inference - CPU (recent: {}ms): {}ms, wall: {}ms (recent: {}ms)".format(
                round(self.clock_times[1] * 1000, 3), round(self.recents_c[1] * 1000, 3), round(self.wall_times[1] * 1000, 3), round(self.recents_w[1] * 1000, 3)))
            print("Average time of postprocessing - CPU: {}ms (recent: {}ms), wall: {}ms (recent: {}ms)".format(
                round(self.clock_times[2] * 1000, 3), round(self.recents_c[2] * 1000, 3), round(self.wall_times[2] * 1000, 3), round(self.recents_w[2] * 1000, 3)))
            return True
        else:
            return False
